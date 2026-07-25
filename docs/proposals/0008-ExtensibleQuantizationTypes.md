# RFC: Extensible Quantization Type System for ONNX

| | |
|---|---|
| **Authors** | Justin Chu (@justinchuby) |
| **Status** | Draft |
| **Created** | 2025-07-22 |
| **ONNX Issue** | TBD |
| **Spec Impact** | TensorProto, ModelProto (new fields), new IR version |

## Abstract

This RFC proposes an extensible quantization type system that allows models to
declare custom quantized data types without requiring changes to the ONNX
specification. The system separates type declaration (structural metadata in the
model) from type implementation (codec logic in the runtime), enabling the
ecosystem to adopt new quantization formats rapidly while preserving backward
compatibility and model portability.

## Motivation

### Problem

The ONNX specification currently supports quantized types through:
1. Built-in data types (INT4, INT8, UINT8, FLOAT8E4M3FN, etc.)
2. QuantizeLinear/DequantizeLinear (QDQ) operators with these types

Every new quantized format requires:
- A spec PR to add the data type
- Potentially a new opset version for QDQ
- Implementation in every conforming runtime

This process is too slow for the rapidly evolving quantization landscape. In
2024-2025 alone, the community has produced: MXFP4, MXFP6, ternary 1.58-bit
(BitNet), IQ1_S through IQ4_NL (llama.cpp), NF4 (QLoRA), various
vendor-specific formats, and many more.

### Evidence of Ecosystem Pain

- llama.cpp maintains 20+ quantized types through a simple codec pattern, adding
  new types in days rather than months
- ONNX Runtime uses `com.microsoft::MatMulNBits` as a vendor op to work around
  spec limitations
- Model converters (GGUF→ONNX, AWQ→ONNX) must dequantize and re-quantize into
  supported types, losing fidelity
- Hardware vendors ship proprietary quantization formats that cannot be
  represented portably

### Goal

Enable any quantization format to be:
1. **Declared** in an ONNX model (portable description)
2. **Executed** by any conformant runtime (through fallback dequantization)
3. **Accelerated** by EPs with native support (no mandatory dequant)

## Comparison with QDQ (QuantizeLinear / DequantizeLinear)

### What QDQ Can Express

Uniform affine quantization: `DequantizeLinear(x, scale, zp) = (x - zp) * scale`
with built-in data types (INT4/INT8/UINT8/FP8), per-tensor or per-channel.

### What QDQ Cannot Express

1. **Non-linear mappings.** NF4 (QLoRA), IQ4_NL — decode via codebook lookup, not
   `(x-zp)*scale`. No QDQ representation exists.
2. **Non-standard packing.** Base-3 ternary (1 byte = 5 values), bit-interleaved
   importance quant — no matching `data_type` in ONNX.
3. **Multi-field block decode.** K-Quants (Q4_K): one 144-byte super-block with
   nested 6-bit sub-scales decodes 256 values cooperatively. QDQ has no
   "block" concept — it is element-wise.
4. **EP identification.** QDQ requires fragile subgraph pattern matching
   (DQ→MatMul→Q). Graph transformations easily break patterns. Extensible types
   use explicit `quant_type_uri` — structural, not positional.

### Coexistence

- QDQ remains valid indefinitely (no deprecation)
- Both can coexist in one model
- `ConvertQDQToExtensible` converter available (opt-in)
- Old runtimes reject extensible-type models via IR version gate

## Design

### Principle: Declaration ≠ Implementation

The model file contains only a **structural declaration** of the quantized type —
enough information for any runtime to produce correct results through a naive
dequantization path. Optimized implementations live in the runtime, not the model.

This is analogous to how video containers declare codec identifiers (H.264, AV1)
without embedding the decoder. Playback software resolves the implementation;
fallback to software decode is always available.

### New Proto Messages

#### QuantTypeDeclProto

Added to `ModelProto.quant_type_declarations` (repeated):

```protobuf
message QuantTypeDeclProto {
  // Globally unique identifier with namespace prefix.
  // Reserved prefixes: "onnx:" (spec-blessed), "vendor:<name>:" (vendor).
  string type_uri = 1;

  // Block structure
  int32 block_size = 2;           // logical elements per block
  int32 bytes_per_block = 3;      // storage bytes per block

  // Encoding descriptor (hints for runtime fast-path selection)
  EncodingDescriptorProto encoding = 4;

  // Scale/zero-point composition
  optional int32 scale_data_type = 5;      // TensorProto.DataType enum
  optional int32 zero_point_data_type = 6;
  int32 group_size = 7;                     // 0 = per-tensor scale

  // Dequantization function — an inline ONNX FunctionProto.
  // Signature: input "packed" (uint8[bytes_per_block]),
  //            input "scale" (scale_data_type scalar)
  //         -> output "values" (float32[block_size])
  // Runtime uses encoding.family as a hint to select native kernels;
  // falls back to executing this function when no native path exists.
  FunctionProto dequant_function = 8;

  // Reference test vector for correctness validation
  // Runtime MAY validate its codec against these vectors.
  bytes test_vector_packed = 9;    // input: one block of packed bytes
  bytes test_vector_float32 = 10;  // expected output: block_size × f32 (IEEE 754 LE)
  float test_vector_scale = 11;    // scale used for test vector
  float test_vector_zero_point = 12;

  // Metadata
  string version = 13;            // semver; semantic changes require version bump
  string description = 14;

  // Partial group handling
  PaddingMode padding_mode = 15;
}

message EncodingDescriptorProto {
  EncodingFamily family = 1;

  // Family-specific parameters (hints for runtime optimization)
  int32 packing_base = 2;          // PACKED_INTEGER: radix (3=ternary, 5=quinary)
  int32 elements_per_unit = 3;     // PACKED_INTEGER: elements encoded per byte/unit
  BitOrder bit_order = 4;
  repeated float codebook = 5;     // LOOKUP_TABLE: fixed codebook values
  float value_offset = 6;          // additive offset after raw decode

  // Nested block structure (K-Quants, MXFP shared exponent, etc.)
  optional NestedBlockLayoutProto nested = 7;
}

message NestedBlockLayoutProto {
  int32 super_block_size = 1;        // total elements per super-block (e.g., 256)
  int32 sub_block_size = 2;          // elements per sub-block (e.g., 32)
  int32 sub_blocks_per_super = 3;    // super / sub

  repeated BlockFieldProto super_fields = 4;  // decoded once per super-block
  repeated BlockFieldProto sub_fields = 5;    // decoded once per sub-block
}

message BlockFieldProto {
  string name = 1;       // referenced in dequant_function graph
  int32 data_type = 2;   // TensorProto.DataType
  int32 bits = 3;        // for sub-byte fields (e.g., 6-bit scales)
  int32 count = 4;       // elements of this field per block (default 1)
}

enum EncodingFamily {
  ENCODING_AFFINE = 0;             // (q - zp) * scale
  ENCODING_SYMMETRIC = 1;          // q * scale
  ENCODING_LOOKUP_TABLE = 2;       // codebook[index] * scale
  ENCODING_PACKED_INTEGER = 3;     // base-N packing
  ENCODING_LOGARITHMIC = 4;       // sign * scale * base^exp
  ENCODING_CUSTOM = 15;            // no standard pattern; rely on dequant_function
}

enum BitOrder {
  BIT_ORDER_LSB_FIRST = 0;
  BIT_ORDER_MSB_FIRST = 1;
}

enum PaddingMode {
  PADDING_ERROR = 0;
  PADDING_ZERO = 1;
  PADDING_REPEAT_LAST = 2;
}
```

#### TensorProto Extension

```protobuf
enum DataType {
  // ... existing values ...
  CUSTOM_QUANT = 32;  // Opaque quantized storage; see quant_type_uri for semantics
}

message TensorProto {
  // ... existing fields ...

  // When set, raw_data contains opaque packed bytes for this quantized type.
  // data_type MUST be set to CUSTOM_QUANT (32).
  // Shape represents logical element counts (not storage bytes).
  // Storage size is determined by: ceil(product(shape) / block_size) × bytes_per_block
  // where block_size and bytes_per_block come from the corresponding QuantTypeDeclProto.
  optional string quant_type_uri = 20;
}
```

**Storage size computation:** For a tensor with logical shape `[M, N]` and a
`QuantTypeDeclProto` with `block_size=B` and `bytes_per_block=P`, the expected
`raw_data` size is `ceil(M × N / B) × P` bytes. Runtimes MUST validate this
before processing.

**Dequant function:** Each `QuantTypeDeclProto` embeds a `dequant_function`
(standard ONNX `FunctionProto`) that defines the canonical dequantization logic
using existing ONNX ops. Runtime resolution order:
1. Native kernel (EP recognizes `encoding.family` + parameters) → fastest
2. Execute the inline `dequant_function` → always correct, portable

The function signature is fixed:
- Input `packed`: uint8[bytes_per_block] — one block of raw packed bytes
- Input `scale`: scalar of `scale_data_type`
- Output `values`: float32[block_size]

This eliminates the need for external codec plugins or a separate formula DSL.
All dequantization knowledge is self-contained in the model.

### Activation Quantization

For static (calibrated) activation quantization, a new repeated field on `GraphProto`:

```protobuf
message GraphProto {
  // ... existing fields ...
  repeated ActivationQuantPolicyProto activation_quant_policies = 20;
}

message ActivationQuantPolicyProto {
  string type_uri = 1;
  ActivationQuantGranularity granularity = 2;
  TensorProto scale = 3;
  TensorProto zero_point = 4;

  // Edge identification
  string producer_output = 5;   // output name of the producing node
  string consumer_input = 6;    // input name of the consuming node
}

enum ActivationQuantGranularity {
  ACTIVATION_QUANT_PER_TENSOR = 0;
  ACTIVATION_QUANT_PER_CHANNEL = 1;
  ACTIVATION_QUANT_PER_TOKEN = 2;
}
```

Dynamic activation quantization requires no model annotation — it is a pure
runtime/EP optimization.

### Runtime Behavior (Informative, not normative)

This section describes expected runtime behavior. ONNX spec normatively defines
only the model format; runtime behavior is implementation-defined but SHOULD
follow these guidelines:

1. **Load model** → parse `QuantTypeDeclProto` list
2. **For each tensor with `quant_type_uri`:**
   - Look up declaration by URI
   - Validate `raw_data` size matches `ceil(elements / block_size) × bytes_per_block`
3. **Execution resolution order:**
   1. EP native kernel (recognizes `encoding.family` + parameters)
   2. User-registered codec (runtime plugin API — see below)
   3. Inline `dequant_function` execution (always available)
4. **Op integration:** A new `DequantizeExtensible` op accepts `CUSTOM_QUANT`
   tensors and outputs float. EPs MAY fuse `DequantizeExtensible → MatMul`
   patterns into quantized compute kernels.

#### User-Registered Codecs (Runtime Extension Point)

Runtimes SHOULD expose an API for users to register high-performance codec
implementations for specific `type_uri` values:

```cpp
// Example C++ API (runtime-specific, not part of ONNX spec)
runtime.RegisterQuantCodec(
    "onnx-community:fp6-llm-e3m2/v1",
    [](const uint8_t* packed, size_t block_size, float scale, float* output) {
        // Custom high-performance dequant kernel
    }
);
```

This allows hardware vendors and researchers to ship optimized kernels without
modifying the model or the runtime source. The inline `dequant_function` in the
model serves as the correctness reference; registered codecs MUST produce
bit-identical results (validated against `test_vector_*` fields).

### DequantizeExtensible Op

A new ONNX operator that bridges `CUSTOM_QUANT` tensors to standard float ops:

```
DequantizeExtensible(input: T1, scale: T2, zero_point?: T2) → output: T3

Type constraints:
  T1: CUSTOM_QUANT
  T2: float16, bfloat16, float32
  T3: float16, bfloat16, float32

Attributes:
  block_size: int (from QuantTypeDeclProto)
```

Semantics: Reads `quant_type_uri` from the input tensor, looks up the
`QuantTypeDeclProto`, and applies dequantization (via native kernel, registered
codec, or inline function).

**Fusion pattern:** EPs SHOULD recognize and fuse:
```
DequantizeExtensible(W) → MatMul(X, W_dequant)
→ fused: QuantizedMatMul(X, W_packed)
```

This preserves compatibility with existing ops while enabling quantized fast paths.

### Backward Compatibility

- Models without `quant_type_uri` are unaffected
- QDQ operators remain valid and unchanged; no deprecation
- Old runtimes reject models with extensible types via IR version check
- A model MAY contain both QDQ patterns and extensible type tensors

### Model Portability Guarantee

Every model using extensible types is guaranteed portable if:
1. `encoding.family != ENCODING_CUSTOM`, AND
2. `test_vector_packed` / `test_vector_float32` are provided

Any conformant runtime can auto-derive a correct (if slow) codec from the formula
and validate it against test vectors. The `ENCODING_CUSTOM` family explicitly
trades portability for expressiveness — models using it acknowledge the runtime
dependency.

### Type URI Governance

| Prefix | Authority | Process |
|--------|-----------|---------|
| `onnx:` | ONNX Steering Committee | Spec PR required |
| `vendor:<name>:` | Named vendor | Self-registered, no approval needed |
| (unprefixed) | Community | No governance, no stability guarantees |

URI format: `<namespace>:<type-name>/v<version>`

Examples:
- `onnx:int4-symmetric/v1`
- `onnx:mxfp4-block32/v1`
- `vendor:qualcomm:ai100-nf4/v2`
- `ggml:iq2_xs/v1`

Versions are immutable. Semantic changes require a new version string.

## Examples

### Example 1: INT4 Symmetric (GPTQ/AWQ-style, QDQ-equivalent)

```
type_uri: "onnx:int4-symmetric/v1"
block_size: 32
bytes_per_block: 18            // 2 (fp16 scale) + 16 (4-bit × 32)
encoding: { family: ENCODING_SYMMETRIC, bit_order: BIT_ORDER_LSB_FIRST }
group_size: 32
scale_data_type: FLOAT16
dequant_function: { // equivalent ops: [UNPACK, CAST(f16), MULTIPLY(scale)] }
```

### Example 2: NF4 (QLoRA / bitsandbytes)

Non-linear 4-bit with 16-value codebook. **Not expressible in QDQ.**

```
type_uri: "onnx-community:nf4/v1"
block_size: 64
bytes_per_block: 34
encoding: {
  family: ENCODING_LOOKUP_TABLE
  codebook: [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
             0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
}
group_size: 64
scale_data_type: FLOAT16
dequant_function: { // equivalent ops: [UNPACK, LOOKUP(codebook), CAST(f16), MULTIPLY(scale)] }
```

### Example 3: Q4_K (llama.cpp K-Quant, nested super-blocks)

256-weight super-block with nested 6-bit sub-block scales. **Not expressible in QDQ.**

```
type_uri: "ggml:q4_k/v1"
block_size: 256
bytes_per_block: 144
encoding: {
  family: ENCODING_AFFINE
  nested: {
    super_block_size: 256
    sub_block_size: 32
    sub_blocks_per_super: 8
    super_fields: [ {name:"d", data_type:FLOAT16}, {name:"dmin", data_type:FLOAT16} ]
    sub_fields: [ {name:"scale", bits:6}, {name:"min", bits:6} ]
  }
}
group_size: 256
dequant_function: { // equivalent ops:
  steps: [UNPACK, CAST(f32), MULTIPLY(sub.scale), MULTIPLY(super.d),
          SUBTRACT(sub.min * super.dmin), CAST(f16)]
}
```

### Example 4: MXFP4 (OCP Microscaling, nested shared exponent)

```
type_uri: "onnx:mxfp4-block32/v1"
block_size: 32
bytes_per_block: 20
encoding: {
  family: ENCODING_AFFINE
  nested: {
    super_block_size: 32
    sub_block_size: 8
    sub_blocks_per_super: 4
    sub_fields: [ {name:"shared_exp", data_type:UINT8, bits:8} ]
  }
}
dequant_function: { // equivalent ops: [UNPACK, CAST(f16), MULTIPLY(sub.shared_exp)] }
```

### Example 5: MXFP6 E3M2 (OCP Microscaling FP6)

6-bit floating point with shared E8M0 block exponent (MX format family).
Same structure as MXFP4, wider elements for better accuracy. **Not expressible in QDQ.**

```
type_uri: "onnx:mxfp6-e3m2-block32/v1"
block_size: 32
bytes_per_block: 25          // 32 × 6 bits = 24 bytes data + 1 byte shared_exp
encoding: {
  family: ENCODING_AFFINE
  bits: 6
  bit_interpretation: FP_E3M2  // 1 sign + 3 exp + 2 mantissa
  nested: {
    super_block_size: 32
    sub_block_size: 32
    sub_blocks_per_super: 1
    sub_fields: [ {name:"shared_exp", data_type:UINT8, bits:8} ]
  }
}
dequant_function: { // equivalent ops: [UNPACK, FP_DECODE(e3m2), CAST(f16), MULTIPLY(sub.shared_exp)] }
```

### Example 6: FP6 LLM (DeepSpeed TC-FPn split storage)

6-bit floating point with split-byte packing for Tensor Core alignment.
Requires custom dequant kernel due to non-standard memory layout. **Requires plugin.**

```
type_uri: "onnx-community:fp6-llm-e3m2/v1"
block_size: 128
bytes_per_block: 112
encoding: {
  family: ENCODING_CUSTOM
  // 6-bit values split into 2-bit + 4-bit segments for TC tile alignment
}
group_size: 128
scale_data_type: FLOAT16
```

### Example 7: 1.58-bit Ternary (BitNet b1.58)

Base-3 packed. 5 values per byte. **Not expressible in QDQ.**

```
type_uri: "onnx-community:ternary-1.58bit/v1"
block_size: 5
bytes_per_block: 1
encoding: {
  family: ENCODING_PACKED_INTEGER
  packing_base: 3
  elements_per_unit: 5
  value_offset: -1.0
}
group_size: 64
scale_data_type: FLOAT16
dequant_function: { // equivalent ops: [UNPACK, ADD(-1.0), CAST(f16), MULTIPLY(scale)] }
test_vector_packed: <0xA4>   // byte 164 = 2*81+0*27+1*9+2*3+2 → [2,2,1,0,2]
test_vector_float32: <...>   // [1,-1,0,-1,1] * 0.5
test_vector_scale: 0.5
```

### Example 8: IQ4_NL (Non-Linear 4-bit, fixed codebook)

```
type_uri: "ggml:iq4_nl/v1"
block_size: 32
bytes_per_block: 18
encoding: {
  family: ENCODING_LOOKUP_TABLE
  codebook: [-1.27, -0.9834, -0.7852, -0.6187, -0.4702, -0.3320, -0.2000, -0.0710,
             0.0710, 0.2000, 0.3320, 0.4702, 0.6187, 0.7852, 0.9834, 1.27]
  bit_order: BIT_ORDER_LSB_FIRST
}
group_size: 32
scale_data_type: FLOAT16
dequant_function: { // equivalent ops: [UNPACK, LOOKUP(codebook), CAST(f16), MULTIPLY(scale)] }
```

### Example 9: IQ1_S (ENCODING_CUSTOM — requires plugin)

```
type_uri: "ggml:iq1_s/v1"
block_size: 256
bytes_per_block: 50
encoding: { family: ENCODING_CUSTOM }
group_size: 256
scale_data_type: FLOAT16
// No formula. Requires codec plugin: "cargo add onnx-codec-ggml"
test_vector_packed: <50 bytes>
test_vector_float32: <256 × f32>
```

### Format Coverage Summary

| Format | bpw | Family | Nested | Native Fast-Path | QDQ-Expressible |
|--------|-----|--------|--------|------------|------------------|
| INT4 Symmetric | 4.5 | SYMMETRIC | No | ✅ | ✅ |
| NF4 (QLoRA) | 4.5 | LOOKUP_TABLE | No | ✅ | ❌ |
| MXFP4 | 5.0 | AFFINE | Yes | ✅ | ❌ |
| Q4_K | 4.5 | AFFINE | Yes (2-level) | ✅ | ❌ |
| Q2_K | 2.625 | AFFINE | Yes (2-level) | ✅ | ❌ |
| IQ4_NL | 4.5 | LOOKUP_TABLE | No | ✅ | ❌ |
| IQ1_S | 1.56 | CUSTOM | N/A | ❌ | ❌ |
| Ternary 1.58 | 1.63 | PACKED_INTEGER | No | ✅ | ❌ |
| FP8 E4M3 | 8.0 | SYMMETRIC | No | ✅ | ✅ |
| AQLM 2×8 | 3.0 | CUSTOM | N/A | ❌ | ❌ |
| MXFP6 E3M2 | 6.125 | AFFINE | Yes | ✅ | ❌ |
| FP6 LLM (TC-FPn) | 6.125 | CUSTOM | No | ❌ | ❌ |

## Impact Assessment

### What changes in the ONNX spec
- New `CUSTOM_QUANT` data type enum value (additive)
- New `DequantizeExtensible` operator (additive)
- New proto messages (additive)
- New optional field on TensorProto (additive)
- New optional field on GraphProto (additive)
- New IR version (gating mechanism)

### What does NOT change
- Existing operators (including QDQ)
- Existing data types
- Existing model semantics
- Backward compatibility with all existing models

### Runtime implementer burden
- **Minimum:** parse new fields + execute inline `dequant_function` for any unrecognized type
- **Recommended:** pattern-match `encoding.family` to dispatch native kernels for common families (AFFINE, SYMMETRIC, LOOKUP_TABLE)
- **Advanced:** EP negotiation, native kernels, plugin registry

## Alternatives Considered

### 1. Keep extending QDQ with new data types
- **Pro:** No schema changes needed
- **Con:** Quadratic spec growth; cannot keep up with research pace

### 2. Embed codec logic (WASM/bytecode) in model
- **Pro:** Fully self-contained models
- **Con:** Security nightmare; model files become executable; massive complexity
- **Resolution:** Model-embedded ONNX *functions* achieve self-containment without
  executable bytecode. The function is a standard ONNX graph — no new security surface.

### 3. Opaque vendor ops (status quo workaround)
- **Pro:** Works today
- **Con:** Zero portability; fragments ecosystem; models locked to one runtime

### 4. This proposal (declarative + inline dequant function)
- **Pro:** Portable, extensible, secure, self-contained, backward compatible.
  Models carry their own dequant logic as standard ONNX ops — no external
  plugins or new ABIs needed. Runtimes optimize via `encoding.family` hints.
- **Con:** New proto fields; function execution slower than native kernels
  (but serves as correctness reference, not hot path)

## Open Questions

1. Should `test_vector_*` fields be mandatory or recommended?
2. ~~Should we define a standard WASM ABI for portable codec plugins?~~
   **Resolved:** The inline `dequant_function` (standard `FunctionProto`) provides
   portability without external plugins or new ABIs.
3. Should `ActivationQuantPolicyProto` live on edges (GraphProto level) or as node
   attributes?
4. Is there value in a "type alias" mechanism (e.g., `onnx:int8-symmetric/v1`
   desugars to a built-in type for migration)?
5. ~~Should the model-embedded dequant function signature be standardized?~~
   **Resolved:** Fixed signature defined in `QuantTypeDeclProto.dequant_function`:
   inputs `packed` (uint8[bytes_per_block]) + `scale` (scale_data_type scalar) →
   output `values` (float32[block_size]).

## References

- [llama.cpp quantization types](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.h)
- [OCP Microscaling (MX) Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [BitNet: 1.58-bit LLMs](https://arxiv.org/abs/2402.17764)
- [ONNX IR spec](https://onnx.ai/onnx/repo-docs/IR.html)
- [ONNX QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)
