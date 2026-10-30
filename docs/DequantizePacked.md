<!--
Copyright (c) ONNX Project Contributors
Copyright (c) 2023-2026 The ggml authors

SPDX-License-Identifier: Apache-2.0 AND MIT
-->

# DequantizePacked decoding specification

This document defines the packed layouts, arithmetic, element ordering, and
codebooks for every format supported by `DequantizePacked` in opset 29.
The signature, storage rules, layouts, algorithms, and codebooks below are
normative. Source provenance and validation notes are informative.

The [operator schema](Operators.md#DequantizePacked) registers this contract.

## Signature and storage

```text
y = DequantizePacked<format, shape, output_dtype=FLOAT>(x)
```

`x` is a rank-one UINT8 tensor of exact payload bytes. The three attributes
describe fixed weight metadata: the quantization format, the logical ONNX
output shape, and the ONNX floating-point output type. `format` and `shape`
are required attributes. `output_dtype` defaults to FLOAT and also supports
FLOAT16, BFLOAT16, and DOUBLE. Blocks are decoded in input order to FLOAT.
Their concatenated values are reshaped to `shape` and converted once to
`output_dtype`. FLOAT16 and BFLOAT16 conversion uses round-to-nearest,
ties-to-even, as in Cast.

`format` is an ONNX-defined, case-sensitive string identifier. The `gguf:`
prefix identifies the encoding family. Each permitted identifier denotes the
fixed layout, algorithm, and codebooks defined in this document. For example:

```text
y = DequantizePacked<format="gguf:q2_0", shape=[64]>(x)
```

The importer selects each tensor's payload byte range, normalizes multibyte
fields to little-endian, and supplies dimensions in ONNX order. In GGML the
first dimension is contiguous; ONNX's last dimension is contiguous. Thus GGUF
dimensions `[ne0, ne1, ne2]` map to ONNX shape `[ne2, ne1, ne0]`.
The flattened element order is preserved.

With `B` elements and `S` bytes per block, require
`len(x) = product(shape[:-1]) * (shape[-1] / B) * S` and `shape[-1] % B = 0`.
Shape has rank at least one, with nonnegative literal dimensions. A zero
dimension produces an empty tensor. The innermost dimension remains
block-aligned for empty outputs.

## Source provenance (informative)

The formats originate from **GGML
[`da8e2a54b919557b816b1c2beb432ccb766f7ce0`](https://github.com/ggml-org/ggml/tree/da8e2a54b919557b816b1c2beb432ccb766f7ce0)**,
whose `GGML_QNT_VERSION` is 2. A GGUF container version, a quantization version,
a model-level `general.file_type`, and a tensor-level `ggml_type` are different
concepts. Importers map GGML's numeric tensor type IDs to the ONNX identifiers
in the table below: for example, GGML type 2 maps to `"gguf:q4_0"`.
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

## Quantization formats

The permitted `format` identifiers, decoded element counts `B`, and packed
byte counts `S` are:

| Identifier | Format  | B   | S   |
| ---------- | ------- | --- | --- |
| gguf:q4_0 | Q4_0 | 32 | 18 |
| gguf:q4_1 | Q4_1 | 32 | 20 |
| gguf:q5_0 | Q5_0 | 32 | 22 |
| gguf:q5_1 | Q5_1 | 32 | 24 |
| gguf:q8_0 | Q8_0 | 32 | 34 |
| gguf:q8_1 | Q8_1 | 32 | 36 |
| gguf:q2_k | Q2_K | 256 | 84 |
| gguf:q3_k | Q3_K | 256 | 110 |
| gguf:q4_k | Q4_K | 256 | 144 |
| gguf:q5_k | Q5_K | 256 | 176 |
| gguf:q6_k | Q6_K | 256 | 210 |
| gguf:q8_k | Q8_K | 256 | 292 |
| gguf:iq2_xxs | IQ2_XXS | 256 | 66 |
| gguf:iq2_xs | IQ2_XS | 256 | 74 |
| gguf:iq3_xxs | IQ3_XXS | 256 | 98 |
| gguf:iq1_s | IQ1_S | 256 | 50 |
| gguf:iq4_nl | IQ4_NL | 32 | 18 |
| gguf:iq3_s | IQ3_S | 256 | 110 |
| gguf:iq2_s | IQ2_S | 256 | 82 |
| gguf:iq4_xs | IQ4_XS | 256 | 136 |
| gguf:iq1_m | IQ1_M | 256 | 56 |
| gguf:tq1_0 | TQ1_0 | 256 | 54 |
| gguf:tq2_0 | TQ2_0 | 256 | 66 |
| gguf:mxfp4 | MXFP4 | 32 | 17 |
| gguf:nvfp4 | NVFP4 | 64 | 36 |
| gguf:q1_0 | Q1_0 | 128 | 18 |
| gguf:q2_0 | Q2_0 | 64 | 18 |

Q8_1 and Q8_K also serve as intermediate/dot-product formats. Q8_1 uses a FLOAT16
scale and 32 signed INT8 quants; Q8_K uses a FLOAT scale and 256 signed INT8
quants. Both decode as `d * q`, with cached sum fields ignored.

MXFP4/NVFP4 payloads interleave scales and quants in blocks with format-specific
byte and lane ordering.

## Packed layouts and decoding

Each block starts at byte offset zero, with the following fields concatenated
in order. `h` denotes an IEEE 754 binary16, `f` an IEEE 754 binary32,
`u8`/`u16` unsigned 8-/16-bit integers, and `i8`/`i16` signed two's-complement
8-/16-bit integers. All multibyte fields are little-endian. Bracketed lengths
count entries of the indicated type. The algorithms below define the bit-plane
and decoded element order.

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
The complete IQ tables are defined in the codebook section below.
Seven stored sign bits are completed with an eighth parity bit for
IQ2_XXS, IQ2_XS, and IQ3_XXS.

NVFP4 scale byte `0x7f` decodes to zero.
MXFP4 multiplies the half scale defined below by doubled E2M1 table values.
Computing the half scale first preserves finite FLOAT values at the largest
exponent, where the full E8M0 scale would overflow.

## Decoding algorithms

The following Python notation defines one block's output as an ordered list of
`B` FLOAT values. `b` is the block's sequence of `S` unsigned bytes, and `kind`
is its format name from the identifier table. Indexing starts at zero; slices
exclude the upper endpoint. Integer arithmetic is exact, `//` is floor division,
and `>>`, `<<`, `&`, and `|` operate on integer bits.

Each floating-point addition, subtraction, and multiplication is rounded
independently to IEEE 754 binary32, round-to-nearest, ties-to-even. Subnormals
use gradual underflow. Overflow produces the corresponding signed infinity.
Signed zero, infinity, and NaN arithmetic follows IEEE 754; NaN payload bits
are unspecified. `mul`, `add`, and `sub` below express these rounding points.

### Common operations

`H` converts a little-endian binary16 field exactly to FLOAT; `R` reinterprets
a 32-bit pattern as FLOAT. `planes` enumerates byte groups in order, then
bit planes from least significant to most significant, then bytes within
each group. `signs` maps each bit to a sign; its parity mode appends an
eighth bit equal to the parity of the seven stored bits.

```python
import math
import struct


def F(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def mul(a, b):
    return F(float(a) * float(b))


def add(a, b):
    return F(float(a) + float(b))


def sub(a, b):
    return F(float(a) - float(b))


def U(b, offset, count=2):
    return sum(b[offset + j] << (8 * j) for j in range(count))


def H(b, offset=0):
    return struct.unpack_from("<e", bytes(b), offset)[0]


def R(bits):
    return struct.unpack("<f", bits.to_bytes(4, "little"))[0]


def signed8(value):
    return value - 256 if value & 128 else value


def planes(b, bits, group):
    return [
        (b[start + j] >> shift) & ((1 << bits) - 1)
        for start in range(0, len(b), group)
        for shift in range(0, 8, bits)
        for j in range(group)
    ]


def signs(value, parity=False):
    if parity:
        value |= (value.bit_count() & 1) << 7
    return [1 - 2 * ((value >> j) & 1) for j in range(8)]


def grid(name, index):
    width, values, bits, text = CODEBOOKS[name]
    packed = bytes.fromhex(text)
    return [
        values[
            (packed[((index * width + j) * bits) // 8]
             >> (((index * width + j) * bits) % 8)) & ((1 << bits) - 1)
        ]
        for j in range(width)
    ]


IQ4 = [-127, -104, -83, -65, -49, -35, -22, -10,
       1, 13, 25, 38, 53, 69, 89, 113]
FP4 = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]
```

### Conventional, K-quant, nonlinear, ternary, and block-floating formats

Offsets passed to `H` and `U`, and offsets in slices, are byte offsets in `b`.
Q8_1's cached `s` and Q8_K's cached `sums` fields do not enter the calculation.
MXFP4 and NVFP4 use the doubled E2M1 values in `FP4` with a half scale; this
also defines scale conversion for every possible scale byte.

```python
def decode_block(b, kind):
    if kind in ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS",
                "IQ3_S", "IQ1_S", "IQ1_M"):
        return decode_iq(b, kind)

    if kind in ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ4_NL"):
        affine = kind in ("Q4_1", "Q5_1")
        offset = 4 if affine else 2
        if kind in ("Q5_0", "Q5_1"):
            high = U(b, offset, 4)
            q = planes(b[offset + 4:], 4, 16)
            q = [q[j] | (((high >> j) & 1) << 4) for j in range(32)]
        else:
            q = planes(b[offset:], 4, 16)
        d = H(b)
        if kind == "IQ4_NL":
            return [mul(d, IQ4[v]) for v in q]
        if affine:
            return [add(mul(d, v), H(b, 2)) for v in q]
        bias = 16 if kind == "Q5_0" else 8
        return [mul(d, v - bias) for v in q]

    if kind in ("Q8_0", "Q8_1", "Q8_K"):
        offset = 2 if kind == "Q8_0" else 4
        count = 256 if kind == "Q8_K" else 32
        d = R(U(b, 0, 4)) if kind == "Q8_K" else H(b)
        return [mul(d, signed8(b[offset + j])) for j in range(count)]

    if kind == "Q2_K":
        q = planes(b[16:80], 2, 32)
        d, m = H(b, 80), H(b, 82)
        return [
            sub(mul(mul(d, b[j // 16] & 15), q[j]),
                mul(m, b[j // 16] >> 4))
            for j in range(256)
        ]

    if kind == "Q3_K":
        q = planes(b[32:96], 2, 32)
        mask = planes(b[:32], 1, 32)
        low = planes(b[96:104], 4, 8)
        high = planes(b[104:108], 2, 4)
        scales = [(low[j] | (high[j] << 4)) - 32 for j in range(16)]
        return [
            mul(mul(H(b, 108), scales[j // 16]), q[j] - 4 * (1 - mask[j]))
            for j in range(256)
        ]

    if kind in ("Q4_K", "Q5_K"):
        s = b[4:16]
        scales = [
            s[j] & 63 if j < 4 else (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4)
            for j in range(8)
        ]
        minima = [
            s[j + 4] & 63 if j < 4 else (s[j + 4] >> 4) | ((s[j] >> 6) << 4)
            for j in range(8)
        ]
        q = planes(b[16 if kind == "Q4_K" else 48:], 4, 32)
        if kind == "Q5_K":
            high = planes(b[16:48], 1, 32)
            q = [q[j] | (high[j] << 4) for j in range(256)]
        d, m = H(b), H(b, 2)
        return [
            sub(mul(mul(d, scales[j // 32]), q[j]), mul(m, minima[j // 32]))
            for j in range(256)
        ]

    if kind == "Q6_K":
        low = planes(b[:128], 4, 64)
        high = planes(b[128:192], 2, 32)
        return [
            mul(mul(H(b, 208), signed8(b[192 + j // 16])),
                (low[j] | (high[j] << 4)) - 32)
            for j in range(256)
        ]

    if kind == "IQ4_XS":
        high = U(b, 2)
        scales = [
            (((b[4 + j // 2] >> (4 * (j % 2))) & 15)
             | (((high >> (2 * j)) & 3) << 4)) - 32
            for j in range(8)
        ]
        q = planes(b[8:], 4, 16)
        return [
            mul(mul(H(b), scales[j // 32]), IQ4[q[j]]) for j in range(256)
        ]

    if kind == "TQ1_0":
        q = [
            (((b[start + j] * 3**plane) & 255) * 3) // 256 - 1
            for start, width, count in ((0, 32, 5), (32, 16, 5), (48, 4, 4))
            for plane in range(count)
            for j in range(width)
        ]
        return [mul(H(b, 52), v) for v in q]

    if kind == "TQ2_0":
        return [mul(H(b, 64), v - 1) for v in planes(b[:64], 2, 32)]

    if kind == "MXFP4":
        e = b[0]
        d = R((0x00200000 << e) if e < 2 else ((e - 1) << 23))
        return [mul(d, FP4[v]) for v in planes(b[1:], 4, 16)]

    if kind == "NVFP4":
        y = []
        for group in range(4):
            e = b[group]
            exponent, mantissa = (e >> 3) & 15, e & 7
            if e in (0, 127):
                d = 0.0
            elif exponent == 0:
                d = mantissa * 2.0**-10
            else:
                d = (1 + mantissa / 8) * 2.0**(exponent - 8)
            q = planes(b[4 + 8 * group:12 + 8 * group], 4, 8)
            y.extend(mul(d, FP4[v]) for v in q)
        return y

    if kind == "Q1_0":
        d = H(b)
        return [d if (b[2 + j // 8] >> (j % 8)) & 1 else -d
                for j in range(128)]

    if kind == "Q2_0":
        return [mul(H(b), v - 1) for v in planes(b[2:], 2, 1)]

    raise ValueError("Unknown format")
```

### IQ2, IQ3, and IQ1 formats

These formats produce eight groups of 32 values per block. Each group consists
of four consecutive eight-value lanes. IQ3 lanes concatenate two four-value
codebook rows. In IQ1_M, four upper nibbles of the scale words form the
binary16 block scale; their remaining bits encode four three-bit local scales
per word. IQ1_S and IQ1_M add the indicated signed `0.125` grid shift before
multiplying by the local scale.

```python
def decode_iq(b, kind):
    d = H(b)
    if kind == "IQ1_M":
        words = [U(b, 48 + 2 * j) for j in range(4)]
        scale_bits = sum((words[j] >> 12) << (4 * j) for j in range(4))
        d = H(scale_bits.to_bytes(2, "little"))
    y = [0.0] * 256
    for group in range(8):
        for lane in range(4):
            i = 4 * group + lane
            if kind == "IQ2_XXS":
                aux = U(b, 6 + 8 * group, 4)
                index = b[2 + 8 * group + lane]
                sign = signs((aux >> (7 * lane)) & 127, parity=True)
                scale = mul(mul(d, 0.5 + (aux >> 28)), 0.25)
                values = grid("iq2xxs_grid", index)

            elif kind == "IQ2_XS":
                word = U(b, 2 + 2 * i)
                sign = signs(word >> 9, parity=True)
                local = (b[66 + group] >> (4 * (lane // 2))) & 15
                scale = mul(mul(d, 0.5 + local), 0.25)
                values = grid("iq2xs_grid", word & 511)

            elif kind == "IQ2_S":
                index = b[2 + i] | (((b[66 + group] >> (2 * lane)) & 3) << 8)
                sign = signs(b[34 + i])
                local = (b[74 + group] >> (4 * (lane // 2))) & 15
                scale = mul(mul(d, 0.5 + local), 0.25)
                values = grid("iq2s_grid", index)

            elif kind == "IQ3_XXS":
                aux = U(b, 66 + 4 * group, 4)
                sign = signs((aux >> (7 * lane)) & 127, parity=True)
                scale = mul(mul(d, 0.5 + (aux >> 28)), 0.5)
                values = (grid("iq3xxs_grid", b[2 + 2 * i])
                          + grid("iq3xxs_grid", b[3 + 2 * i]))

            elif kind == "IQ3_S":
                indices = [
                    b[2 + 2 * i + j]
                    | (((b[66 + group] >> (2 * lane + j)) & 1) << 8)
                    for j in range(2)
                ]
                sign = signs(b[74 + i])
                local = (b[106 + group // 2] >> (4 * (group % 2))) & 15
                scale = mul(d, 1 + 2 * local)
                values = grid("iq3s_grid", indices[0]) + grid("iq3s_grid", indices[1])

            elif kind == "IQ1_S":
                high = U(b, 34 + 2 * group)
                index = b[2 + i] | (((high >> (3 * lane)) & 7) << 8)
                scale = mul(d, 2 * ((high >> 12) & 7) + 1)
                delta = -0.125 if high & 0x8000 else 0.125
                values = [add(v, delta) for v in grid("iq1s_grid", index)]
                sign = [1] * 8

            elif kind == "IQ1_M":
                high = b[32 + i // 2] >> (4 * (i % 2))
                index = b[i] | ((high & 7) << 8)
                local = (words[group // 2]
                         >> (6 * (group % 2) + 3 * (lane // 2))) & 7
                scale = mul(d, 2 * local + 1)
                delta = -0.125 if high & 8 else 0.125
                values = [add(v, delta) for v in grid("iq1s_grid", index)]
                sign = [1] * 8

            for j in range(8):
                value = mul(scale, values[j])
                y[8 * i + j] = (value if kind in ("IQ1_S", "IQ1_M")
                                else mul(value, sign[j]))
    return y
```

## IQ codebooks

Each codebook entry below is `(width, values, bits, hex_bytes)`. Concatenate
the adjacent hexadecimal string literals and read each pair of hex digits
as one byte. Successive `bits`-wide fields, least-significant field first
within each byte, index the signed integer list `values`. Group these decoded
entries into rows of `width` values. Rows are in increasing codebook-index
order. The complete row counts are 256 for `iq2xxs_grid`, 512 for
`iq2xs_grid`, 1024 for `iq2s_grid`, 256 for `iq3xxs_grid`, 512 for `iq3s_grid`,
and 2048 for `iq1s_grid`. The `grid` helper implements this definition.

<!-- BEGIN GENERATED GGUF CODEBOOKS -->

```python
CODEBOOKS = {
    "iq2xxs_grid": (
        8,
        (8, 25, 43),
        2,
        (
            "00000200050008000a00110014002000220028002a00410044005000580061006400800082008a00a2000101"
            "0401100115014001840198010002020222028202010404041004210424044004420448046004810484049004"
            "a404000502050805200546056905800591050906100640068406a40600080508080814082808410844085008"
            "5208880804094009020a140a01100410101021104010601084109010951000110811201150115a1180112412"
            "451200140814201425144914801418156215001616160118041810184018811800190519a019511a00200220"
            "0a2044206120802082202921482100220222012404241024402456240025412564259026082820289428442a"
            "0140044010401840214024404040484056406040814084409040004120416141804185410142104248425642"
            "684200440844204480449944124524450046014804481048404845480049584961498249454a904a00500850"
            "1150195020508050885004514251a4519152905492540a550156545600581158195864584059085a04601060"
            "406068600061556118626062006405641065126584654268008002800a804180828004811881408111820184"
            "0484108415844084608400854685948509864086608602880489118a0490109024904090a190169180914592"
            "00942294449451958198209902a050a085a009a100a218a450a804a9"
        ),
    ),
    "iq2xs_grid": (
        8,
        (8, 25, 43),
        2,
        (
            "00000200050008000a0011001400160019002000220025002800410044004600490050005200550058006100"
            "64008000820085008800910094009900a000010104010601090110011201150118011a012101240140014201"
            "4501480151015401600168018101840190010002020205020802110214022002410244025002550280028a02"
            "0104040406040904100412041504180421042404400442044504480451045404560460048104840490040005"
            "0205050508051105140520054105440550056105800501060406100626064006420684060008020805080808"
            "0a08110814082008250841084408500858088008a008aa08010904091009400981098909000a200a280a960a"
            "a00a011004100610091010101210151018102110241040104210451048105110541060106a10811084109010"
            "0011021105110811111114112011411144115011801194119611011204120612101240126012001402140514"
            "0814111414142014411444144914501464148014011504151015401500161416491601180418101812184018"
            "54188618001905196619511aa91a00200220052008200a201120142020204120442050208020a02001210421"
            "1021402148216521002222228022a82201240424102429244024002541255225992501261a26a62600280828"
            "0a28202855288828a22868299029082a202a822a882a8a2a0140044006400940104012401540184021402440"
            "40404240454048404a4051405440604065408140844090400041024105410841114114412041414144415041"
            "80418541a2410142044210421242294240420044024405440844114414441944204441444444504480449444"
            "014504451045244540459a4500460a4644465046014804481048404845485448624800491149444950496949"
            "044a00500250055008501150145020502850415044505050805001510451105115514051425100524452aa52"
            "0154045410542154405460548154a154005508558055885521566856a156005814584158505899581a594059"
            "4259855a0160046010604060546062608660a960006124624a62926200641664106540654565a46501686a68"
            "2569066a546a626a00800280058008801180148020802a8041804480508080808280a880aa80018104810681"
            "10814081518159810082208280828282a082a8820184048410841284158440846084898400854485a5851886"
            "6a860088088825885a8880888288a8880689228a808a888a968aa88a01900490109040905690849000912291"
            "6491569289920094059444945094589429959095929541965198a6984999159a609a00a002a008a00aa020a0"
            "2aa0a0a051a159a1a6a100a202a208a22aa280a2a0a240a495a465a698a60aa820a822a828a8a0a8a8a804a9"
            "84a986a928aa2aaa91aaaaaa"
        ),
    ),
    "iq2s_grid": (
        8,
        (8, 25, 43),
        2,
        (
            "00000200050008000a0011001400160019002000220025002800410044004600490050005200550058006100"
            "640066006900800082008500880091009400a000a500aa000101040106010901100112011501180121012401"
            "4001420145014801510154015601590160016501680181018401900192019501a101a4010002020205020802"
            "1102140220022a02410244024602490250025502800285028a029402a2020104040406040904100412041504"
            "1804210424042604290440044204450448044a04510454045604590460046204650481048404860489049004"
            "95049804a104a40400050205050508050a051105140516051905200525052805410544054605490550055205"
            "5505580561056405800582058505880591059405a00501060406060609061006150640064506480651065406"
            "600681068406900600080208050808081108140816081908200825082a084108440846084908500852085508"
            "580861086408800885089408aa08010904091009120915091809210940094509480951095409600981099009"
            "000a110a140a220a280a2a0a500a990a01100410061009101010121015101810211024102610401042104510"
            "481051105410561059106010621065106810811084108610901095109810a110a41000110211051108110a11"
            "1111141116111911201122112511281141114411461149115011521155115811611164118011821185118811"
            "9111941101120412091210121512211224124012451251125412811284129012001402140514081411141414"
            "161419142014251428144114441446144914501452145514581461146414801482148514881491149414a014"
            "0115041506150915101512151515181521152415401542154515481551155415601581158415901500160516"
            "08161116141620164116441650168016aa160118041806180918101815181818211840184218451848185118"
            "541860188118841800190219051908191119141920194119441950196919a219041a101a401a561a00200220"
            "052008201120142016201920202025202a2041204420502052205520642080208a209420aa20012104211021"
            "12211521212140214221452151215421602181218421902100220a22222228222a224422502288228a22a822"
            "0124042406240924102415241824212424244024422445244824512454246024812484249024002505250825"
            "112514252025412544255025662580250126042610264026592600280528112814284128442850288a28aa28"
            "01290429102995290a2a222a642a882a8a2a014004400640094010401240154018401a402140244026404040"
            "4240454048404a40514054405640594060406240654081408440904095409840a140a4400041024105410841"
            "1141144116411941204122412541414144414641494150415241554158416141644180418241854188419141"
            "9441a04101420442104212421542184224424042454248425142544260428142844200440244054408440a44"
            "1144144416441944204422442544284441444444464449445044524455445844614464448044824485448844"
            "91449444a044014504450645094510451245154518452145244540454245454548455145544560456a458145"
            "8445904500460246054608461146144620464146444650468046a54601480448094810481248154818482148"
            "2448404842484548484851485448604884489048004902490549084911491449204941494449504980499649"
            "014a044a104a404a005002500550085011501450165019502050225025502850415044504650495050505250"
            "5550585061506450805082508550885091509450015104510651095110511251155118512151245140514251"
            "4551485151515451605181518451905100520552085211521452205241524452505269528052015404540654"
            "0954105412541554185421542454405442544554485451545454605481548454905400550255055508551155"
            "1455205541554455505580550156045610562656405600580258055808581158145820584158445850585a58"
            "80580159045910594059005a195a855aa85a0160046006601060126015601860216024604060456048605160"
            "54606060846090600061026105610861116114612061416144615061806199610462106240625662a1620064"
            "05640864116414642064416444645064806401650465106540654a6568659265006694660168046810686568"
            "986800692a69426aa16a00800280058008801180148019802080258041804480508052805580588061808080"
            "858091809480018104810981108112811581188121812481408142814581488151815481818184819081a981"
            "008205820a821182148241824482508201840484068409841084128415841884218440844284458448845184"
            "54846084818484849084008502850585088511851485208541854485508580858a8501860486108629864086"
            "0088058811881488418844885088a2880189048940896589228a588a5a8a828aa28a01900490099010901290"
            "159018902490409042904590489051905490609081908490909000910591119114914191449150915a910192"
            "049210924092a69200940294059408941194149420944194449450948094969401950495109540959895a195"
            "00964696649601980498109826984098a998009949995299909a00a005a00aa014a022a02aa041a044a050a0"
            "a2a0aaa040a165a102a20aa222a228a22aa282a288a28aa2a8a201a404a410a440a489a4a4a400a519a551a6"
            "0aa828a8a2a854a986a908aa0aaa20aa22aa28aa88aaaaaa"
        ),
    ),
    "iq3xxs_grid": (
        4,
        (4, 12, 20, 28, 36, 44, 52, 62),
        4,
        (
            "0000020004001100130017002000220031004200730075000101030110011201210125013001320141015401"
            "7001000202020402110220022202310233023702510257027502010307031003120325037003130437044404"
            "5704730475040105070532055205350664061007140716074307610701100310101012102110231030103210"
            "3410471050100011021111112011221101120312101212122112301272120013021320133113461366130114"
            "0514501420152415461571150516221740170020022011201320202022202620312042200121032105211021"
            "1221212130216321672170210022022211221722202222223722402255220123102314237023742335245324"
            "0325272541257425012703271627452701301030123021302330503065307230003102312031313144314631"
            "0132033210322532523272321133333330344734723400350635223555351436363663363337603704401740"
            "3540374053405740744120423742404260426642074345430444514464442545434570450547104712473047"
            "1250415070500051065126515551145232527252025353531054235427547254025531555056245742572460"
            "4460466064602161616117626462306336634464056552653366036721670370057007701070327052702671"
            "40711272457252720073157333736073217441740075027524753076"
        ),
    ),
    "iq3s_grid": (
        4,
        (1, 3, 5, 7, 9, 11, 13, 15),
        4,
        (
            "0000010002000500070010001100120014001600200021002500330040004200450047005100530060006200"
            "7100740077000001010102010401100111011501200123012701310135014401610165017201000201020502"
            "0702100213021602210225023002340242024502470251025302700273020303110315032003220331033303"
            "3603440350035203670371037503000413041704210424043204400443045104700402050405200522052605"
            "3305410545054705660573050606110613063106520671060007020704072007220726073307500754070010"
            "0110021004101010111013101510171020102210311034103610541056106110721000110111031106111011"
            "1411211130113311411150115211701176110012121215121712201224123212401243125512601272120113"
            "0413071310131313211327133013341341136213701303140514121414143114331442144614501454140115"
            "1015131521153015321551152016241627164416461601170317101712172117351741176217701700200120"
            "0320052007201020122014201620212023202720302032204120432045205020522067207020732075200021"
            "0221102113211721222125213121342142215121012204220722212223223022372241225322572271227422"
            "0023022305231123222324233123332342235023662301240724202423243224352441247224752404251125"
            "2225372540255325702500260226072621265526612605271127262730274327502702301130133015301730"
            "2230313033303530423044304730513063307130013103310531143121312331403160317231763100321232"
            "2032323234325032013310331433213323332733303341334333473355337333033411341634223431345234"
            "6034643401351035123525353235443556357335163641360137033720372237353700400440124020402440"
            "2740324041405040704002410741114113412241304135414341514155410142034210421542214233424042"
            "5742624270420443114313432043224331433543004402442444374440447144054507452145624513463446"
            "6046104715473047434751470250105014502250405044504750525066507450015103510551125121513251"
            "7251005211522352305236525352025307531053275344535153655373530154045420543254465412552655"
            "5155535542560257045722571160136015603160336060600061206127616461126234624262556262627062"
            "0063146321634063256443646264006503653465606505664066116713670070047007702070227036704070"
            "5470627002711171247143714571017204721072167221723072517202733273357353730174057413742074"
            "507422754275027631760077"
        ),
    ),
    "iq1s_grid": (
        8,
        (0, 1, -1),
        2,
        (
            "aaaaa9aaa0aaa6aaa5aa88aa80aa9aaa99aa96aa95aa20aa08aa02aa01aa10aa6aaa69aa66aa65aa40aa5aaa"
            "59aa56aa55aaa2a8a0a888a882a881a884a885a890a828a821a824a809a800a805a818a812a811a816a860a8"
            "48a842a841a850a8aaa9a9a9a6a9a5a980a99aa999a996a995a920a908a904a912a914a96aa969a966a965a9"
            "48a940a944a95aa959a956a955a988a282a281a290a228a224a200a205a212a210a248a244a250a2a8a0a2a0"
            "a0a0a1a080a086a085a094a02aa020a025a00aa008a002a000a001a004a01aa019a010a016a015a068a048a0"
            "40a046a045a058a052a050a051a054a082a184a128a122a10aa109a100a106a11aa118a111a114a160a148a1"
            "42a144a1aaa6a9a6a6a6a5a680a69aa699a696a695a620a608a601a610a66aa669a666a665a640a65aa659a6"
            "56a655a6a0a488a482a484a492a490a428a40aa408a400a418a412a414a448a442a441a444a450a4aaa5a9a5"
            "a6a5a5a580a59aa599a596a595a520a508a504a518a510a56aa569a560a566a565a540a55aa559a556a555a5"
            "8a8a888a828a848a928a908a288a228a0a8a008a068a188a128a108a148a488a428a418a588a508aa888a288"
            "a188a4888a88898880888688988892889488208825880a880888098802880088018804881a88108862884988"
            "40885888528888898289818990892a892189248909890089068905891289118960894889428941895089a882"
            "a182a48282828082868284829882918228822082218226822582088202820082018204821982108216826282"
            "64824a82428240824682448245825882528250825482a980a080a580888082808080818084809a8099809080"
            "9680958028802280208021800880098002800080018004800580188012801080118014806a80698062806080"
            "668065804a804880428040804180448045805a8059805080a881a281a081a181808181818681858198819181"
            "2a81298122812081268125810881008101810681048118811281108116811481158161816581498140815281"
            "548188868186908628862286218624860a860086068605861a86188612861186148660864886428650868a84"
            "8984808485849884908429842284208426840884028400840184048405841a84108415846484488449844084"
            "4684588451845484a4858185928591852285218524850a850985008506851885118514856085488541854585"
            "aa9aa99aa69aa59a809a9a9a999a909a969a959a209a089a049a189a109a6a9a699a669a659a409a5a9a599a"
            "509a569a559aa0988898829884989098299822982498009806980598189812981098119860984a9841984498"
            "5098a899a699a599889980999a9999999699959920990899019904991099689966996599489940995a995999"
            "56995599a0928292819284929092229220922192249209920092069205921192609248924292449258925092"
            "a4908090989094902a9020902690089002900090049019901090169064904a90429040904690459058905290"
            "51905490a0918a918991849190912891249100911a9118911491629161914a914591aa96a996a696a5968096"
            "9a9699969696959620960896029610966a966996669665965a96599656965596a49488948294849490942194"
            "249409940094189412941194149460944a944194449452945094aa95a995a695a5959a959995969595952095"
            "08950195049510956a9569956695659540955a95599556955595a02a882a812a902a242a092a002a062a052a"
            "182a122a112a422a442a582a512aaa28a828a228a128a4288928802881288628852898289128942820282628"
            "25280828022800280128042805281028162815286828622861284a28492840285a2858285928a02988298229"
            "81299029282909290029052912291429642942295029a8228022842294222022262225220822022200220122"
            "182219221022162215226822612264224a22492240225a2258225422a820a920a020a5208820822080208120"
            "84209a2090209520282022202020212024200a20082002200020012006200420182012201020112014206920"
            "622060206620482042204020412044204520502056205520a821a021a4218221802186218521982192219421"
            "2a212921202126210a210821092100210121042119211021162168216021652142214021582152215121a026"
            "8826802685269026292624260a26002606261826122611261426602648264226412644265026a824a024a124"
            "a5248a248224802486249824922491242a24202425240824092402240024012404241a241924102411241524"
            "612464244924402441244624582452245124542481252225212524250025062505251225142542255025a80a"
            "a20aa00aa10aa40a890a800a850a980a920a940a2a0a200a260a080a020a000a010a040a100a160a610a640a"
            "400a460a5a0a580a510a540aa008a608a408a5088808820880088108860884089a0890089108960895082808"
            "22082008210824080a0808080908020800080108060804080508180812081008110814086908600848084208"
            "4008410844085a0850085508a809a10989098009850998099209290920092509080902090009010904091909"
            "100960094a0949094009440945095209a202a002880282028002810286028402980290029602950228022202"
            "20022102240225020a0208020202000201020602040205021802190212021002110214026a02660265024802"
            "42024002410244025802520250025502a800a900a200a000a100a4008a008800890082008000810084008500"
            "980092009000910094002a0028002900220020002100260024000a0008000900020000000100060004000500"
            "1a001800120010001100160014001500680062006000640065004a0048004200400041004600440058005200"
            "500051005400aa01a801a901a201a101a601a401880182018001860184019a01980199019201900191019601"
            "94012801200121012601240125010a0108010901020100010101060104010501180112011001140169016001"
            "610166016401650148014001450159015001510156015401a206a006a106a4068a0680068606980695062006"
            "260625060806020600060106060604061a06190612061006690664064a0649064006460658065406a804a904"
            "a004a504880482048004810484049004280422042004210424040a0408040904020400040104060404040504"
            "180412041004110414046804600464044804420440044104460444045004a205a605800585059a0590059105"
            "940520052605240508050005010506050405190510051605150568056505490540054105460545055805a01a"
            "821a811a841a901a221a0a1a001a011a061a051a181a121a111a141a681a411a501aa818a218a118a4188918"
            "8018981899189118941820182418081800180118041810181118151862186518491840185818511854188819"
            "811984192a19281921190019011906191a196019481941195019881289128012811285129812911294122a12"
            "29122012261225120812021200120112041205121a12191210126212601264124a1249124212401241124612"
            "4512581252125412a010a610a510881080108110841022102010211024100a10081002100010011004101810"
            "1210101011101410611064106510481040104110441045105910501051105610a911a41180119a1191119611"
            "94112a112011261125110811021100110111061105111a11101116116a116911601165114211411146114411"
            "5a115211511155118116841690162816091600160516181614166016481646165116a814a2148a1480149814"
            "9214911494142a142814201421142614081402140014011404141a1410141514691462146514401458145214"
            "501454148815811586152815221524150a1500150615051512151015141561154215461545155115aa6aa96a"
            "a66aa56a9a6a996a966a956a206a0a6a086a026a016a046a106a6a6a696a666a656a406a5a6a596a566a556a"
            "a068886882688168846890682868226824680a68096800680168066804681268116814686068646842684168"
            "44685068aa69a969a669a56980699a6999699669956908690269046910696a6969696669656940695a695969"
            "56695569826284622862226208620062056218621262146242624462a860a460896080608560916094602a60"
            "286020602660086002600060016004600560106011601660156068606260616064604a604960406046605160"
            "886181618461906128612261246125610a610061046105611861116115616061486145615261aa66a966a666"
            "a56680669a6699669666956628662066086602660466106614666a6669666666656640665a66596656665566"
            "a064a164886482648164906428642264216424640a64096400640564186412646064416444645064aa65a965"
            "a665a56580659a6599659665956520650865026501656a6569656665656540655a65596556655565a04a884a"
            "814a864a844a904a284a214a244a004a064a054a144a154a604a484a424a414a444a504aa848a248a148a448"
            "8a4880488648854898489248914894482a4820480a4808480248004801480448194810486248614849484048"
            "46485848524851485448a04988498249844990492249214924490a4909490049064911491449604942494149"
            "5449a842a242a1428a428042864291422a4225420842024200420142064204421a4218421942104262426142"
            "494242424042464258425442aa40a040a640a5408a4088408240804081408440984090409440954028402240"
            "2040214024400a40084009400240004001400640044005401840124010401140144068406040664048404940"
            "424040404140444045405a405940504056405540a841a2418a41804184419a41914194412041264124410841"
            "0941004101410441104116416941624164416541494142414041524151415441a04681468446904628462146"
            "0a4609460046014605461246104660464846414644465046a244a144a4448a4489448044864485449a449844"
            "924491442a442944204426442544084402440044014404441944104411441544684462444a44494440444544"
            "58445144a045804590452245214524450a4500450645184560454845424540454145aa5aa95aa65aa55a805a"
            "9a5a995a965a955a205a085a025a015a045a6a5a695a665a655a405a5a5a595a565a555aa058a45888588258"
            "8158845885582158245808580058065805581858125860584a58495841584458a959a659a5598a5984599959"
            "96599559205908590159045910596a5969596659655940595a5959595659555984529052285222520a520252"
            "005206520552185210521152165214526052a150a4508a508950805086509150945029502050085002500050"
            "015004501050155068506250605061506450495040504650a051885181518551985190512251215125510951"
            "0051015106511a51195161514a51405141514451585152515151aa56a956a656a5569a569956965695560856"
            "0256015604566a5669566656655640565a56595656565556a054825484549854905428540a54005405541854"
            "115414544a544154aa55a955a655a5559a559955965595550855025501556a5569556655655540555a555955"
            "56555555"
        ),
    ),
}
```

<!-- END GENERATED GGUF CODEBOOKS -->

## Reference implementation and validation (informative)

The reference implementation uses NumPy and ONNX's dtype support to decode
one block at a time.

The IQ tables are compacted without loss from the pinned source. Regenerate
both the reference implementation's tables and this document's codebook
section with:

```text
pixi run python tools/gen_gguf_codebooks.py <path-to-pinned-ggml-common.h>
```

The generator checks the input file's SHA256 before extracting numeric tables.
The upstream MIT notice is retained next to the generated data in
`onnx/reference/ops/GGML_LICENSE`.

`tests/python/dequantize_packed_test.py` constructs a deterministic 64-block
nonzero corpus for each format, including negative/zero/subnormal scales,
high-bit masks, signs, group scales, and varied codebook indices. Expected
FLOAT output digests were obtained from the pinned GGML C row decoders,
compiled with strict floating-point arithmetic; Q8_1 uses the documented
`d*q` fields because upstream has no row decoder for it. Complete codebook digests
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
It also executes the normative algorithms in this document and compares their
FLOAT bits directly with the upstream C results.
Normal tests compare results against the committed digests.

The tests also cover shape/order, all four output types, default output type,
empty tensors, malformed lengths and metadata, rejected enums, ignored cached
sums, and every MXFP4/NVFP4 scale encoding. Backend examples have hand-written
expected results for Q4_0, Q5_1, IQ4_NL, MXFP4, and an empty K-quant tensor.
The documented algorithms are checked for every format, every complete IQ
codebook, every MXFP4/NVFP4 scale byte with all E2M1 values, and IEEE scale
encodings including signed zero, subnormals, infinities, and NaNs.
Shape inference validates all format sizes, unknown packed lengths, static
output dimensions, block alignment, and byte-count overflow.

## License notice (informative)

The following notice applies to the GGML-derived decoding logic and IQ codebooks.

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
