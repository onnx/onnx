# RFC: Extensible Quantization Type System for ONNX

| | |
|---|---|
| **Authors** | Justin Chu (@justinchuby) |
| **Status** | Draft |
| **Created** | 2025-07-22 |
| **ONNX Issue** | TBD |
| **Spec Impact** | TensorProto, TypeProto, ModelProto (new fields), new IR version |

## Abstract

This RFC proposes an open-world quantization type system for ONNX. A model can
carry an opaque quantized tensor, identify its format with an immutable URI, and
reference a model-local ONNX function that defines the canonical decoding
semantics. Runtimes may dispatch optimized kernels by URI or execute the
referenced decoder as a portable fallback.

The IR is extended once to introduce this mechanism. New quantization formats
thereafter require only a new URI and decoder; they do not require new
`TensorProto.DataType` values, protobuf fields, or IR-version bumps.

## Motivation

### Problem

ONNX currently represents quantization through:

1. Built-in data types such as INT4, INT8, FLOAT4E2M1, and FLOAT8E4M3FN.
2. QuantizeLinear and DequantizeLinear (QDQ) operators over supported types.

Every fundamentally new packed format may require a specification change,
release, and runtime update before a conformant model can use it. This does not
scale to formats such as NF4, K-Quants, IQ types, AQLM, Sherry/STQ1_0, and
vendor-specific hardware layouts.

The logical tensor shape is also insufficient to determine storage size for
these formats. A payload may contain sub-byte values, multiple interleaved bit
streams, embedded scales, sparse indices, codebooks, padding, or alignment.

### Goals

Enable a quantization format to be:

1. **Represented** without adding a new ONNX data type for every format.
2. **Self-contained** through a portable decoder expressed with ONNX operators.
3. **Accelerated** by runtimes with a native implementation for the exact URI.
4. **Preserved** by tools that do not recognize the format.
5. **Validated** structurally without teaching the checker every format.

### Non-goals

- Replacing QDQ or existing built-in low-precision types.
- Standardizing optimized kernels or runtime plugin APIs.
- Defining activation-quantization policy annotations.
- Making existing ONNX operators accept opaque quantized values.
- Encoding every possible quantization family as a protobuf `oneof`.

## Comparison with QDQ

QDQ is appropriate for supported element types and quantization equations,
especially uniform affine quantization:

```text
DequantizeLinear(x, scale, zero_point) = (x - zero_point) * scale
```

It does not directly describe:

- Codebook mappings such as NF4 and IQ4_NL.
- Multi-field block payloads such as K-Quants.
- Non-standard packing and element placement such as STQ1_0.
- Formats whose storage includes sparse indices or other block metadata.

QDQ remains valid and unchanged. Models may use both QDQ and extensible
quantized tensors.

## Design Principles

### Open-world identity

Each format has an immutable URI. The URI identifies complete decoding
semantics, including packing, byte order, blocking, padding, and logical element
placement. If any semantic aspect changes, the producer uses a new URI.

The URI, rather than an `EncodingFamily` enum, identifies the format.
Optimized dispatch uses both the URI and a digest of the authoritative decoder,
so a model cannot bind a familiar URI to different semantics. Consequently, a
new format does not require changing the ONNX IR.

### Decoder is authoritative

Every portable declaration references a model-local `FunctionProto`. The
function defines how the opaque bytes become a logical FLOAT tensor. Optional
layout metadata supports validation and introspection, but never replaces or
changes the decoder semantics.

### Declaration is separate from implementation

The model declares the format and its decoder. A runtime may:

1. Execute a native fused kernel recognized by exact URI and decoder digest.
2. Execute another implementation known to conform to that pair.
3. Invoke the referenced model-local decoder.

Optimized implementations are runtime concerns. The model does not contain
native code, WASM, or another executable codec format.

## Proposed IR Additions

The field numbers and the numeric value of `CUSTOM_QUANT` are placeholders
until implementation.

### FunctionIdentifierProto

Model-local functions are already identified by `(domain, name, overload)`.
Quantization declarations reference a function rather than embedding a second
copy of it.

```protobuf
message FunctionIdentifierProto {
  string domain = 1;
  string name = 2;
  string overload = 3;
}
```

### QuantizedStorageLayoutProto

The layout is optional, generic metadata for regular block formats. It can
describe byte and bit ranges without attempting to classify the quantization
algorithm.

```protobuf
message QuantizedFieldProto {
  string name = 1;

  // Location and repetition within one storage block, in bits.
  uint64 bit_offset = 2;
  uint32 bit_width = 3;
  uint64 count = 4;
  uint64 bit_stride = 5;

  // Logical interpretation when the field is directly representable by an
  // ONNX scalar type. UNDEFINED means that the decoder interprets the bits.
  int32 data_type = 6;
}

message QuantizedStorageLayoutProto {
  uint64 values_per_block = 1;
  uint64 bytes_per_block = 2;
  repeated QuantizedFieldProto field = 3;
}
```

`QuantizedFieldProto` describes physical storage only. It intentionally does
not define codebook lookup, scaling, sparsity, or output permutation. Those are
semantic operations expressed by the decoder. Future formats that do not fit a
regular block layout omit this message and remain fully representable.

### QuantizationTestVectorProto

Test vectors help runtime implementers validate optimized decoders. They are
not a substitute for the decoder and are not required for model execution.

```protobuf
message QuantizationTestVectorProto {
  bytes payload = 1;
  repeated int64 logical_dims = 2;
  int32 block_axis = 3;
  repeated float decoded_float32 = 4 [packed = true];
}
```

### QuantTypeDeclProto

Declarations are stored in `ModelProto.quantization_type_declarations`.

```protobuf
message QuantTypeDeclProto {
  // Immutable identity for the complete decoding semantics.
  string type_uri = 1;

  // Authoritative portable decoder.
  FunctionIdentifierProto decoder = 2;

  // SHA-256 digest of the deterministic protobuf serialization of the
  // referenced FunctionProto. Native dispatch requires an exact match.
  bytes decoder_sha256 = 3;

  // Optional structural information for validation and optimization.
  optional QuantizedStorageLayoutProto storage_layout = 4;

  repeated QuantizationTestVectorProto test_vector = 5;
  string doc_string = 6;

  // Non-semantic metadata. Decoding correctness MUST NOT depend on it.
  repeated StringStringEntryProto metadata_props = 7;
}
```

The URI includes its semantic version. There is no separate version field that
could disagree with the URI.

The digest is computed from the referenced `FunctionProto` using deterministic
protobuf serialization. Decoder functions cannot call model-local functions,
so the digest covers the complete portable semantics. Documentation and
metadata fields participate in the digest; changing them produces a different
implementation identity even when tensor results are unchanged.

### QuantizedTensorInfoProto

The tensor annotation refers to a declaration and defines how regular blocks
are applied to this tensor.

```protobuf
message QuantizedTensorInfoProto {
  string type_uri = 1;

  // Axis containing independent sequences of storage blocks.
  // Negative values are normalized against the logical rank.
  // Required when the declaration has storage_layout.
  optional int32 block_axis = 2;
}
```

### TensorProto and TypeProto extensions

`TensorProto` remains the storage envelope rather than introducing a parallel
tensor message. This preserves existing names, metadata, external-data support,
checksums, and initializer infrastructure.

```protobuf
enum DataType {
  // ... existing values ...
  CUSTOM_QUANT = TBD;
}

message TensorProto {
  // ... existing fields ...
  optional QuantizedTensorInfoProto quantization = TBD;
}

message TypeProto {
  message Tensor {
    // ... existing fields ...
    optional QuantizedTensorInfoProto quantization = TBD;
  }
}
```

For `CUSTOM_QUANT`:

- `TensorProto.dims` is the logical decoded shape.
- Payload bytes are stored in `raw_data` or referenced by `external_data`.
- Typed data fields such as `int32_data` and `float_data` MUST NOT be used.
- `TensorProto.quantization` MUST be present and its URI MUST resolve to exactly
  one declaration.
- `TypeProto.Tensor.quantization` MUST be present when `elem_type` is
  `CUSTOM_QUANT` and MUST be absent for every other element type.
- A `TypeProto.Tensor` annotation corresponding to an initializer MUST agree
  with the initializer annotation.
- `CUSTOM_QUANT` MUST NOT be used as the element type of SparseTensorProto,
  TypeProto.SparseTensor, sequence elements, map values, or optional values in
  this initial proposal.
- Unknown annotations and declarations MUST round-trip without modification.

Adding quantization information to `TypeProto` is necessary because intermediate
values are not represented by `TensorProto`. Operators cannot infer a format by
searching for an initializer with the same name.

## Storage Validation

For a declaration with:

```text
values_per_block = B
bytes_per_block = P
block_axis = A
```

and logical dimensions `D`, blocks do not cross independent sequences along
other axes:

```text
sequence_count = product(D[i] for i != A)
blocks_per_sequence = ceil(D[A] / B)
expected_payload_bytes = sequence_count * blocks_per_sequence * P
```

The checker MUST use overflow-checked arithmetic. The payload length comes from
`len(raw_data)` or the external-data `length` entry. The `length` entry is
required for external `CUSTOM_QUANT` data. If a declaration omits
`storage_layout`, payload size is validated only by the referenced decoder and
external-data bounds.

For regular layouts, `values_per_block` and `bytes_per_block` MUST be greater
than zero. `block_axis` MUST normalize to an axis in the logical rank; scalar
tensors therefore cannot use a regular block layout. Every field MUST have
positive `bit_width`, `count`, and `bit_stride`.

The decoder defines padding and trimming of a final partial block. Producers
SHOULD prefer shapes divisible by `values_per_block` when the format does not
have canonical partial-block semantics.

For each field in a regular storage layout, the checker validates:

```text
bit_offset + (count - 1) * bit_stride + bit_width
    <= bytes_per_block * 8
```

Fields may overlap when the decoder intentionally gives the same bits multiple
interpretations. Layout metadata MUST agree with the authoritative decoder.

## Decoder Contract

The referenced model-local function has this required signature:

```text
Decode(payload: uint8[N], logical_shape: int64[rank], block_axis: int64 scalar)
    -> values: float32[logical_shape]
```

- `payload` is the exact raw or external byte sequence.
- `logical_shape` is the tensor's logical shape.
- `block_axis` is normalized to `[0, rank)`, or is `-1` when no regular storage
  layout is declared.
- The output is FLOAT with that shape.
- Scales, zero points, codebooks, sparse metadata, and padding are either
  encoded in the payload or represented as constants in the function.
- The function MUST have exactly three inputs and one output with the types
  above. Its `value_info` entries MUST provide these types.
- The function MUST NOT declare required attributes.
- The function MUST use only standard-domain ONNX operators from its declared
  opset imports.
- The function MUST NOT call model-local functions or custom-domain operators.
- The function MUST NOT contain control-flow, random, nondeterministic, or
  side-effecting operators.
- The function MUST be deterministic for valid inputs.

The decoder may use operations such as BitShift, BitwiseAnd, Gather,
ScatterElements, Reshape, Transpose, Cast, and BitCast. This provides a bounded,
dataflow-based language for unpacking and decoding without introducing a second
quantization-specific DSL.

The specification maintains a decoder-safe operator allowlist. Expanding that
allowlist or permitting a newer standard opset does not change the IR format,
although runtimes may reject a decoder using operators they do not implement.

A runtime invokes the function through the extensible-quantization mechanism;
the raw byte view is not itself exposed as an ordinary reinterpretation of a
`CUSTOM_QUANT` tensor in the graph.

## DequantizeExtensible Operator

A new operator bridges opaque quantized tensors to standard tensor values:

```text
DequantizeExtensible(X: CUSTOM_QUANT) -> Y: FLOAT
```

The operator:

1. Reads the quantization identity carried by the input type.
2. Resolves the declaration by exact URI.
3. Uses a conforming native implementation or invokes the referenced decoder.

Scale, zero point, block size, and output type are not operator inputs or
attributes because they are format semantics. A model may use `Cast` after
dequantization when another floating-point output type is desired.

Execution providers MAY fuse patterns such as:

```text
DequantizeExtensible(W) -> MatMul(X, W_float)
```

into a native quantized MatMul without materializing `W_float`.

Initially, `DequantizeExtensible` is the only standard operator required to
accept `CUSTOM_QUANT`. Other operators MUST reject it unless their schemas
explicitly define behavior for opaque quantized values. In particular, this RFC
does not add implicit `Gather`, slicing, or reshaping semantics for packed
payloads.

## Runtime Behavior

Expected runtime resolution order:

1. Validate the declaration, tensor annotation, and payload bounds.
2. Look for an optimized implementation registered for the exact
   `(type_uri, decoder_sha256)` pair.
3. Otherwise invoke the referenced model-local decoder.
4. Reject the model if the decoder is missing, invalid, or uses unsupported
   operators.

Optimized implementations conform to the decoder under normal ONNX numerical
accuracy requirements. Bit-identical floating-point results are not required
unless the format specification independently requires them.

Runtime-specific plugin APIs are permitted but are outside the ONNX
specification. A portable model does not depend on a plugin.

## Checker Behavior and Forward Compatibility

The checker does not need built-in knowledge of every URI. It validates:

- URI presence and uniqueness within the model.
- Declaration and function-reference resolution.
- Decoder digest, ABI, operator allowlist, and absence of required attributes.
- `CUSTOM_QUANT` storage-field restrictions.
- Tensor/TypeProto annotation consistency.
- Prohibition of `CUSTOM_QUANT` in unsupported container and sparse types.
- External-data bounds.
- Regular block and field bounds when layout metadata is present.
- Decoder function well-formedness under existing FunctionProto rules.

An unrecognized URI is not an error when its declaration and decoder are
present. Tools that do not execute the model can inspect, copy, rename, and
round-trip the tensor without understanding the format.

## URI Governance and Format Versioning

URI format:

```text
<authority>:<format-name>/v<semantic-version>
```

Examples:

- `onnx:mxfp4-block32/v1`
- `ggml:q4_k/v1`
- `org.tencent.sherry:stq1_0/v1`

Suggested governance:

| Prefix | Authority | Process |
|--------|-----------|---------|
| `onnx:` | ONNX project | Specification review |
| Organization or project namespace | Namespace owner | No ONNX approval required |

The complete decoding semantics of a URI are immutable. A change to field
packing, byte order, scale interpretation, padding, or logical element mapping
creates a new URI version.

This versioning is independent of the ONNX IR version. The IR version changes
only when the extensibility framework itself changes.

## Examples

The examples show declarations conceptually; function bodies are abbreviated.

### INT4 symmetric block quantization

```text
type_uri: "example.org:int4-symmetric-block32/v1"
decoder: ("example.org.quant", "DecodeInt4Block32", "v1")
storage_layout:
  values_per_block: 32
  bytes_per_block: 18
  fields:
    - {name: "values", bit_offset: 0, bit_width: 4,
       count: 32, bit_stride: 4, data_type: UNDEFINED}
    - {name: "scale", bit_offset: 128, bit_width: 16,
       count: 1, bit_stride: 16, data_type: FLOAT16}
```

The decoder extracts signed 4-bit values, bitcasts the embedded FP16 scale, and
multiplies the values by that scale.

### NF4

```text
type_uri: "example.org:nf4-block64/v1"
decoder: ("example.org.quant", "DecodeNF4Block64", "v1")
storage_layout:
  values_per_block: 64
  bytes_per_block: 34
  fields:
    - {name: "indices", bit_offset: 0, bit_width: 4,
       count: 64, bit_stride: 4, data_type: UNDEFINED}
    - {name: "scale", bit_offset: 256, bit_width: 16,
       count: 1, bit_stride: 16, data_type: FLOAT16}
```

The decoder contains the 16-entry NF4 codebook as a Constant, gathers values by
the unpacked indices, and applies the embedded scale.

### Sherry STQ1_0

STQ1_0 demonstrates why the decoder, not a closed family enum, must be
authoritative. One block stores multiple bit streams, an embedded scale, and a
non-contiguous logical element mapping.

```text
type_uri: "org.tencent.sherry:stq1_0/v1"
decoder: ("org.tencent.sherry", "DecodeSTQ1_0", "v1")
storage_layout:
  values_per_block: 256
  bytes_per_block: 42
  fields:
    - {name: "code", bit_offset: 0, bit_width: 4,
       count: 64, bit_stride: 4, data_type: UNDEFINED}
    - {name: "sign", bit_offset: 256, bit_width: 1,
       count: 64, bit_stride: 1, data_type: BOOL}
    - {name: "scale", bit_offset: 320, bit_width: 16,
       count: 1, bit_stride: 16, data_type: FLOAT16}
```

For group `g` in `[0, 64)`, the decoder:

```text
code = extract_4_bits(payload, g)
sign = extract_1_bit(payload, 256 + g)
vector = codebook[16 * sign + code]  // four ternary lanes

chunk = g / 16
lane_in_chunk = g % 16
output_index(p) = 64 * chunk + lane_in_chunk + 16 * p, p in [0, 4)

output[output_index(p)] = vector[p] * scale
```

The codebook is a constant in the decoder. The placement rule captures the
stride-16 grouping used within each 64-value chunk. The resulting storage is 42
bytes for 256 values: 1.25 bits per weight for codes and signs, or 1.3125 bits
per weight including the FP16 block scale.

No new protobuf case or IR version is needed to introduce STQ1_0. Another
format with different fields or element placement supplies a different URI and
decoder.

## Backward Compatibility

- Models without extensible quantized tensors are unaffected.
- QDQ and built-in low-precision types remain unchanged.
- A new IR version gates the initial framework.
- Old runtimes reject models requiring the new IR version.
- New formats using the established framework do not require later IR bumps.

## Impact Assessment

### ONNX specification changes

- One opaque `CUSTOM_QUANT` data-type value.
- Quantization declaration and annotation messages.
- New fields on ModelProto, TensorProto, and TypeProto.Tensor.
- A `DequantizeExtensible` operator.
- One new IR version to gate the framework.

### Runtime implementer burden

Minimum support consists of:

- Parsing and preserving the new fields.
- Validating payload and declaration structure.
- Invoking the referenced model-local decoder for
  `DequantizeExtensible`.

Optimized support consists of dispatching fused kernels by exact URI.

## Alternatives Considered

### Continue adding every format as a built-in type

This gives standardized operator support but requires an ONNX release and IR
change for each new storage type. Built-in types remain appropriate for mature,
widely supported formats, but they are not the only extension mechanism.

### Closed quantization-family protobuf

A `oneof` containing affine, codebook, vector-codebook, sparse, logarithmic,
and block formats is readable for known cases but makes that list the extension
boundary. Formats combining multiple bit streams, embedded metadata, unusual
element placement, or future mechanisms would require new protobuf cases.

This RFC instead uses generic storage metadata plus an authoritative decoder.
Convenience libraries may provide reusable decoder functions for common
families without placing those families in the IR.

### Dedicated QuantizedTensorProto

A separate tensor message clearly distinguishes logical shape from physical
storage, but it would need to duplicate TensorProto naming, metadata,
external-data, graph-attribute, and type-propagation integration.

Extending TensorProto once provides the same logical/storage separation while
reusing existing infrastructure.

### Put all metadata on QDQ nodes

Operator attributes are extensible, but an initializer would not be
self-describing and intermediate quantized values would not carry their format
identity. Keeping identity in TensorProto and TypeProto makes it structural.

### Embed native code, WASM, or custom bytecode

This would make models executable containers and introduce a substantial
security and implementation burden. A standard ONNX function is inspectable,
validated through existing mechanisms, and sufficient as a correctness
fallback.

### Opaque vendor operators only

Vendor operators work today but provide no portable fallback semantics and
lock models to runtimes implementing those operators.

## Open Questions

1. Should regular `storage_layout` metadata be required for ONNX-governed URIs
   even though it remains optional for other formats?
2. Should test vectors be recommended or required for ONNX-governed URIs?
3. Should the first revision require the block axis to be the final logical
   axis to simplify runtime implementations?
4. Which opset version should the decoder function target to balance bit-level
   expressiveness and runtime availability?
5. Should a later proposal define packed-data-preserving operators such as a
   block-aligned Gather?

## References

- [llama.cpp quantization types](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.h)
- [Sherry: Hardware-Efficient 1.25-Bit Ternary Quantization](https://arxiv.org/abs/2601.07892)
- [OCP Microscaling (MX) Formats v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [BitNet: 1.58-bit LLMs](https://arxiv.org/abs/2402.17764)
- [ONNX IR specification](https://onnx.ai/onnx/repo-docs/IR.html)
- [ONNX QuantizeLinear](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)
