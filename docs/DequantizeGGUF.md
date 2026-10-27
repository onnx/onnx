<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# DequantizeGGUF proposal: interoperability notes

`DequantizeGGUF` in development opset 29 represents the decoding of a single
GGUF tensor's packed quantized payload. It allows importers to preserve packed
weights as UINT8 initializers, including external-data initializers, rather
than expanding every weight before exporting an ONNX model. Backends can
constant-fold decoding or recognize a decoding/matrix-multiplication pattern.
The operator does not prescribe fusion or a quantized arithmetic kernel.

The [operator schema](Operators.md#DequantizeGGUF) defines the contract. This
document records the rationale and upstream evidence behind that contract.

## Signature and storage

```text
y = DequantizeGGUF<ggml_type, shape, output_dtype=FLOAT>(x)
```

`x` is a rank-one UINT8 tensor of exact payload bytes. The three attributes
describe fixed weight metadata: the tensor's GGML type ID, the logical ONNX
output shape, and the ONNX floating-point output type. Attributes are preferable
to runtime inputs for metadata that must be known to determine output type and
shape. FLOAT, FLOAT16, BFLOAT16, and DOUBLE outputs are supported; decoding uses
FLOAT arithmetic before the final conversion.

This is deliberately not a file reader. The importer removes GGUF headers and
alignment padding, selects each tensor's byte range, normalizes multibyte
fields to little-endian, and supplies dimensions in ONNX order. In GGML the
first dimension is contiguous; ONNX's last dimension is contiguous. Thus GGUF
dimensions `[ne0, ne1, ne2]` map to ONNX shape `[ne2, ne1, ne0]`, without
reordering the decoded values. Any architecture-specific weight permutation,
global multiplier, or other transformation remains outside the operator.

With `B` elements and `S` bytes per block, require
`len(x) = product(shape[:-1]) * (shape[-1] / B) * S` and `shape[-1] % B = 0`.
No truncated blocks, row padding, extra bytes, inferred dimensions, or implicit
transpose are allowed. Zero dimensions mean empty dimensions, not Reshape-style
copying. The innermost dimension must remain block-aligned for empty outputs.

## Versioned upstream basis

The source of truth is **GGML
[`da8e2a54b919557b816b1c2beb432ccb766f7ce0`](https://github.com/ggml-org/ggml/tree/da8e2a54b919557b816b1c2beb432ccb766f7ce0)**,
whose `GGML_QNT_VERSION` is 2. A GGUF container version, a quantization version,
a model-level `general.file_type`, and a tensor-level `ggml_type` are different
concepts. The operator fixes concrete layouts and codebooks; it does not
dynamically follow an upstream branch or accept arbitrary future enum values.
Older files with different layouts must be converted by the importer.

| Evidence                                                                                                                                     | What it establishes                                                                         |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [GGUF specification](https://github.com/ggml-org/ggml/blob/da8e2a54b919557b816b1c2beb432ccb766f7ce0/docs/gguf.md#file-structure)             | Tensor descriptors, payload offsets, alignment, little- and big-endian files                |
| [`include/ggml.h`](https://github.com/ggml-org/ggml/blob/da8e2a54b919557b816b1c2beb432ccb766f7ce0/include/ggml.h)                            | Stable numeric tensor IDs, removed IDs, quantization version, contiguous dimension ordering |
| [`src/ggml-common.h`](https://github.com/ggml-org/ggml/blob/da8e2a54b919557b816b1c2beb432ccb766f7ce0/src/ggml-common.h)                      | Packed block structs, sizes, IQ lookup tables, Q8 auxiliary fields                          |
| [`src/ggml-quants.c`](https://github.com/ggml-org/ggml/blob/da8e2a54b919557b816b1c2beb432ccb766f7ce0/src/ggml-quants.c)                      | Reference `dequantize_row_*` algorithms                                                     |
| [`src/ggml-impl.h`](https://github.com/ggml-org/ggml/blob/da8e2a54b919557b816b1c2beb432ccb766f7ce0/src/ggml-impl.h)                          | Exact MXFP4 E8M0 and NVFP4 unsigned E4M3 scale decoding                                     |
| [llama.cpp GGUF reader](https://github.com/ggml-org/llama.cpp/blob/19cce6919992fdffd92e0ac152caaf1e2b8f254e/gguf-py/gguf/gguf_reader.py)     | Reversing descriptor dimensions and exposing quantized payloads as UINT8                    |
| [llama.cpp quantization sizes and decoder](https://github.com/ggml-org/llama.cpp/tree/19cce6919992fdffd92e0ac152caaf1e2b8f254e/gguf-py/gguf) | Independent Python definitions of block sizes and decoding families                         |

The GGUF prose enum lags the live GGML header: for example, it does not yet list
NVFP4, Q1_0, and Q2_0 at the pinned revision. Consequently the proposal follows
the header and implementation, not just the prose enum.

## Quantized coverage and exclusions

All **27 active quantized tensor types** are supported: Q4_0, Q4_1, Q5_0, Q5_1,
Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ2_XXS, IQ2_XS, IQ3_XXS,
IQ1_S, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, IQ1_M, TQ1_0, TQ2_0, MXFP4, NVFP4,
Q1_0, and Q2_0. The schema lists every numeric ID and block size.

Q8_1 and Q8_K are primarily intermediate/dot-product formats. They are included
to avoid making the quantized set implicitly incomplete. Q8_1 uses a FLOAT16
scale and 32 signed INT8 quants; Q8_K uses a FLOAT scale and 256 signed INT8
quants. Both decode as `d * q`; cached sums are not additive biases.
Q8_1's older FLOAT-scale layout is not this proposal's layout.

Plain F32, F16, BF16, F64, I8, I16, I32, and I64 are excluded because ONNX
already represents their values directly. Removed Q4_2/Q4_3 and repacked
Q4_0/IQ4_NL IDs 31-33 and 36-38 are also excluded. Runtime-specific interleaved
layouts and fork-specific enums are not standardized by this operator.

MXFP4/NVFP4 remain in scope: although ONNX has FLOAT4E2M1 and low-precision scale
types, these GGML payloads interleave scales and quants in blocks and have
specific byte-order and lane-order conventions. They are not plain FLOAT4
initializers. Filename labels such as `Q4_K_M` denote model-level type mixtures,
not additional tensor types or values of `ggml_type`.

## Packed layouts and decoding

The following fields appear in order without extra padding. `h` denotes a
little-endian FLOAT16, `f` a little-endian FLOAT, `u16` a little-endian unsigned
16-bit integer, and bracketed lengths count entries of the indicated type. Bit-plane order and
codebook indexing follow the pinned row decoders, not generic adjacent-bit
unpacking.

| Format  | Packed fields                               | Decoding summary                                                                        |
| ------- | ------------------------------------------- | --------------------------------------------------------------------------------------- |
| Q4_0    | `h d; u8 q[16]`                             | Scale times signed 4-bit values with offset 8; low nibbles precede high nibbles         |
| Q4_1    | `h d,m; u8 q[16]`                           | `d*q + m` with unsigned nibbles                                                         |
| Q5_0    | `h d; u8 high[4],q[16]`                     | Combine high-bit mask and nibbles, subtract 16, multiply by `d`                         |
| Q5_1    | `h d,m; u8 high[4],q[16]`                   | Unsigned 5-bit `d*q + m`                                                                |
| Q8_0    | `h d; i8 q[32]`                             | `d*q`                                                                                   |
| Q8_1    | `h d,s; i8 q[32]`                           | `d*q`; ignore cached `s`                                                                |
| Q2_K    | `u8 scales[16],q[64]; h d,dmin`             | Per-16-element 4-bit scales/minima; `d*scale*q - dmin*min`                              |
| Q3_K    | `u8 mask[32],q[64],scales[12]; h d`         | Signed 3-bit values with packed signed 6-bit group scales                               |
| Q4_K    | `h d,dmin; u8 scales[12],q[128]`            | Per-32-element 6-bit scales/minima and unsigned nibbles                                 |
| Q5_K    | `h d,dmin; u8 scales[12],high[32],q[128]`   | Q4_K-style scales/minima with unsigned 5-bit quants                                     |
| Q6_K    | `u8 low[128],high[64]; i8 scales[16]; h d`  | Signed 6-bit quants times signed per-16-element scales and `d`                          |
| Q8_K    | `f d; i8 q[256]; i16 sums[16]`              | `d*q`; ignore cached sums                                                               |
| IQ2_XXS | `h d; u16 q[32]`                            | 256-entry 8-value grid, parity-completed signs, per-32-element scale                    |
| IQ2_XS  | `h d; u16 q[32]; u8 scales[8]`              | 512-entry 8-value grid, parity-completed signs, nibble group scales                     |
| IQ2_S   | `h d; u8 q[64],high[8],scales[8]`           | 1024-entry 8-value grid; payload contains indices and explicit sign bytes               |
| IQ3_XXS | `h d; u8 q[96]`                             | 256-entry 4-value grid; packed group scales and parity-completed signs                  |
| IQ3_S   | `h d; u8 q[64],high[8],signs[32],scales[4]` | 512-entry 4-value grid, explicit signs, odd group scales                                |
| IQ1_S   | `h d; u8 q[32]; u16 high[8]`                | 2048-entry 8-value ternary grid, odd scales, signed 0.125 grid shift                    |
| IQ1_M   | `u8 q[32],high[16],scales[8]`               | Same grid with per-16-element scales; FLOAT16 scale bits distributed across scale words |
| IQ4_NL  | `h d; u8 q[16]`                             | Nonlinear 16-value table times `d`                                                      |
| IQ4_XS  | `h d; u16 high; u8 scales[4],q[128]`        | IQ4_NL table with signed 6-bit group scales                                             |
| TQ1_0   | `u8 q[48],high[4]; h d`                     | Base-3 unpacking with wrapped byte arithmetic and specified lane ordering               |
| TQ2_0   | `u8 q[64]; h d`                             | Unpack 2-bit planes, subtract 1, multiply by `d`                                        |
| MXFP4   | `u8 exponent,q[16]`                         | E8M0 block scaling and low/high nibble E2M1 values                                      |
| NVFP4   | `u8 scales[4],q[32]`                        | Four 16-element groups with unsigned E4M3 scales and E2M1 values                        |
| Q1_0    | `h d; u8 q[16]`                             | Adjacent little-bit-order bits select `-d` or `d`                                       |
| Q2_0    | `h d; u8 q[16]`                             | Adjacent 2-bit codes, subtract 1, multiply by `d`                                       |

The IQ4_NL table is
`[-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113]`.
Other IQ tables are fixed by the pinned `ggml-common.h`, not supplied as model
inputs. Seven stored sign bits are completed with an eighth parity bit for
IQ2_XXS, IQ2_XS, and IQ3_XXS.

Scale conversion follows GGML, including its nonstandard encodings:
NVFP4 scale byte `0x7f` decodes to zero, not an ONNX FLOAT8 NaN. MXFP4 uses
`ggml_e8m0_to_fp32_half` multiplied by doubled E2M1 table values. In particular,
the implementation must not first round the full E8M0 scale to FLOAT: at the
largest exponent that would overflow before multiplying by 0.5. The proposal
does not replace these conversions with nominal low-precision ONNX casts.
All 256 possible scale bytes are covered by explicit tests.

## Reference implementation and validation

The implementation depends only on ONNX's existing NumPy and dtype support,
not on a GGUF package, C library, or file parser. It is intentionally a
readable block decoder rather than a performance kernel.

The IQ tables are compacted without loss from the pinned source. Reproduce
them with:

```text
pixi run python tools/gen_gguf_codebooks.py <path-to-pinned-ggml-common.h>
```

The generator checks the input file's SHA256 before extracting numeric tables.
The upstream MIT notice is retained next to the generated data in
`onnx/reference/ops/GGML_LICENSE`.

`tests/python/dequantize_gguf_test.py` constructs a deterministic 64-block
nonzero corpus for each format, including negative/zero/subnormal scales,
high-bit masks, signs, group scales, and varied codebook indices. Expected
FLOAT output digests were obtained from the pinned GGML C row decoders,
compiled with strict floating-point arithmetic; Q8_1 uses the documented
`d*q` fields because upstream has no row decoder for it. No expected digest
is derived from the ONNX implementation. Complete codebook digests independently
cover every lookup entry.

Reproduce the upstream C comparison with an installed C compiler and CMake:

```text
pixi run python tools/verify_gguf_reference.py <path-to-pinned-ggml-src-directory>
```

Only the three pinned source files are required; the script checks their hashes,
extracts the upstream decoder functions unchanged, and builds a temporary oracle
library with strict FLOAT arithmetic. Its Q8_1 oracle is the explicit field
definition above. The tool checks both decoded FLOAT bits and stored test digests,
then verifies every codebook's digest directly from the upstream numeric tables.
Normal tests do not need the upstream source or a C compiler.

The tests also cover shape/order, all four output types, default output type,
empty tensors, malformed lengths and metadata, rejected enums, ignored cached
sums, and every MXFP4/NVFP4 scale encoding. Backend examples have hand-written
expected results for Q4_0, Q5_1, IQ4_NL, MXFP4, and an empty K-quant tensor.
Shape inference validates all format sizes, unknown packed lengths, static
output dimensions, block alignment, and byte-count overflow.

This is an operator proposal for Operators SIG review, not a claim that
existing ONNX runtimes already implement it.
