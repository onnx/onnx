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
embed an ONNX function that defines the canonical decoding semantics. Runtimes
may dispatch optimized kernels by URI or execute the embedded decoder as a
portable fallback. Formats that cannot be expressed by the safe decoder subset
may explicitly rely on a runtime codec plugin.

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
Consequently, a new format does not require changing the ONNX IR.

### Decoder is authoritative

Every portable declaration embeds a `FunctionProto` that defines how the opaque
bytes become a logical FLOAT tensor. The function belongs to the quantization
type declaration; it is not added to `ModelProto.functions` and does not occupy
the model-local function namespace.

Optional descriptor attributes support validation, introspection, plugin
selection, and tooling, but never replace or change the decoder semantics.

### Declaration is separate from implementation

The model declares the format and its decoder. A runtime may:

1. Execute a built-in implementation recognized by URI.
2. Execute an implementation supplied by an optional runtime plugin.
3. Invoke the embedded decoder as the portable fallback.

Optimized implementations are runtime concerns. The model does not contain
native code, WASM, or another executable codec format.

## Proposed IR Additions

The field numbers and the numeric value of `CUSTOM_QUANT` are placeholders
until implementation.

### QuantizationProto

Only one new message type is required. Declarations are stored in
`ModelProto.quantizations`.

```protobuf
message QuantizationProto {
  // Immutable identity for the complete decoding semantics.
  optional string type_uri = 1;

  // Authoritative portable decoder. This function is scoped to this message
  // and is not registered as a model-local function. If absent, the type is
  // opaque and requires a built-in or plugin codec.
  optional FunctionProto decoder = 2;

  // Optional open-ended descriptors. Standard attribute names are defined
  // below, and formats may add namespaced attributes without changing the IR.
  repeated AttributeProto descriptor = 3;

  // Optional reference vector for optimized implementations.
  optional TensorProto test_payload = 4;  // UINT8, one-dimensional payload
  optional TensorProto test_output = 5;   // FLOAT, logical decoded shape
  optional int32 test_block_axis = 6;

  optional string doc_string = 7;

  // Non-semantic metadata. Decoding correctness MUST NOT depend on it.
  repeated StringStringEntryProto metadata_props = 8;
}

message ModelProto {
  // ... existing fields ...
  repeated QuantizationProto quantizations = TBD;
}
```

The URI includes its semantic version. There is no separate version field that
could disagree with the URI.

`descriptor` deliberately reuses `AttributeProto` instead of introducing a
closed hierarchy of layout, field, codebook, sparse, or block messages. The
decoder remains sufficient when no standardized descriptor applies.
Descriptor names MUST be unique within a declaration, and every descriptor
MUST be a concrete attribute with `ref_attr_name` absent.

The following descriptor names are initially standardized for regular block
storage:

| Name | Attribute type | Meaning |
|------|----------------|---------|
| `values_per_block` | INT | Logical values decoded by one block |
| `bytes_per_block` | INT | Physical bytes occupied by one block |
| `field_names` | STRINGS | Names of physical fields |
| `field_layout` | TENSOR of INT64 `[N, 5]` | Rows are `(bit_offset, bit_width, count, bit_stride, data_type)` |

Bit offset zero is the least-significant bit of payload byte zero. The
descriptor describes physical storage only. Codebook lookup, scaling,
sparsity, and output permutation remain in the decoder. Additional attributes
SHOULD use namespaced names such as `org.example.layout_kind`.

The two test tensors reuse `TensorProto` instead of adding a test-vector message.
They MUST either both be present or both be absent. `test_payload` MUST be a
one-dimensional UINT8 tensor, and `test_output` MUST be a FLOAT tensor whose
dimensions are the logical decoded shape. `test_block_axis` follows the same
normalization rules as a tensor annotation and MUST be present exactly when the
test uses regular block descriptors. Test tensors are optional for portable
declarations and SHOULD be present for opaque plugin-only declarations.

### TensorProto and TypeProto extensions

`TensorProto` remains the storage envelope rather than introducing a parallel
tensor message. This preserves existing names, metadata, external-data support,
checksums, and initializer infrastructure.

```protobuf
message TensorProto {
  enum DataType {
    // ... existing values ...
    CUSTOM_QUANT = TBD;
  }

  // ... existing fields ...
  optional string quantization_type_uri = TBD;
  optional int32 quantization_block_axis = TBD;
}

message TypeProto {
  message Tensor {
    // ... existing fields ...
    optional string quantization_type_uri = TBD;
    optional int32 quantization_block_axis = TBD;
  }
}
```

For `CUSTOM_QUANT`:

- `TensorProto.dims` is the logical decoded shape.
- Payload bytes are stored in `raw_data` or referenced by `external_data`.
- Typed data fields such as `int32_data` and `float_data` MUST NOT be used.
- `quantization_type_uri` MUST be present and resolve to exactly one
  `ModelProto.quantizations` entry.
- `quantization_type_uri` and `quantization_block_axis` MUST be absent for every
  other element type.
- A `TypeProto.Tensor` corresponding to an initializer MUST agree with the
  initializer fields.
- `CUSTOM_QUANT` MUST NOT be used as the element type of SparseTensorProto,
  TypeProto.SparseTensor, sequence elements, map values, or optional values in
  this initial proposal.
- Unknown descriptor attributes and declarations MUST round-trip without
  modification.

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
required for external `CUSTOM_QUANT` data. If a declaration omits the
standardized regular-layout descriptors, payload size is validated only by the
embedded decoder or runtime codec and external-data bounds.

When `values_per_block` or `bytes_per_block` is present, both MUST be present
and greater than zero. `quantization_block_axis` MUST normalize to an axis in
the logical rank and MUST be present; scalar tensors therefore cannot use a
regular block layout. `quantization_block_axis` MUST be absent when regular
block descriptors are absent.

`field_names` and `field_layout` MUST either both be present or both be absent,
and they are valid only when the two block-size descriptors are present. Every
`field_layout` row MUST have a nonnegative `bit_offset`, positive `bit_width`,
`count`, and `bit_stride`, and a valid `TensorProto.DataType` value. Its row
count MUST equal the length of `field_names`.

The decoder defines padding and trimming of a final partial block. Producers
SHOULD prefer shapes divisible by `values_per_block` when the format does not
have canonical partial-block semantics.

For each field in a regular storage layout, the checker uses overflow-checked
arithmetic to validate:

```text
bit_offset + (count - 1) * bit_stride + bit_width
    <= bytes_per_block * 8
```

Fields may overlap when the decoder intentionally gives the same bits multiple
interpretations. For portable declarations, descriptor metadata MUST agree with
the authoritative decoder. For opaque declarations, descriptors and codecs
MUST agree with the URI owner's external format specification.

## Decoder Contract

The embedded function has this required signature:

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

The initial decoder profile uses the standard ONNX domain at opset 26 or later
and permits only the following deterministic, dataflow operators:

```text
Constant, ConstantOfShape, Identity,
Cast, CastLike, BitCast,
Add, Sub, Mul, Div, Mod, Pow, Neg, Abs, Min, Max,
BitShift, BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot,
Equal, Greater, GreaterOrEqual, Less, LessOrEqual, Where,
Shape, Size, Range, Reshape, Flatten, Transpose, Squeeze, Unsqueeze,
Concat, Split, Slice, Expand, Tile, Pad,
Gather, GatherElements, GatherND, ScatterElements, ScatterND, NonZero,
ReduceSum, ReduceMin, ReduceMax, MatMul
```

No operator containing a graph attribute is allowed. Expanding this profile or
permitting a newer standard opset does not change the IR format, although older
runtimes may reject a decoder using operators they do not implement.

A runtime invokes the embedded function through the extensible-quantization
mechanism; the raw byte view is not itself exposed as an ordinary
reinterpretation of a `CUSTOM_QUANT` tensor in the graph.

### Portability levels

A declaration with an embedded decoder is **portable**. Any runtime implementing
`DequantizeExtensible` and the decoder's standard operators can execute it,
although a plugin may still provide a faster implementation.

A declaration without an embedded decoder is **opaque**. It is useful for
proprietary formats or codecs that cannot be expressed by the decoder-safe
operator subset, but it requires a matching built-in implementation or plugin.
Runtimes without one MUST reject the model. URIs under the `onnx:` namespace
MUST be portable.

## DequantizeExtensible Operator

A new operator bridges opaque quantized tensors to standard tensor values:

```text
DequantizeExtensible(X: CUSTOM_QUANT) -> Y: FLOAT
```

The operator:

1. Reads the quantization identity carried by the input type.
2. Resolves the declaration by exact URI.
3. Uses a conforming built-in/plugin implementation or invokes the embedded
   decoder.

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
2. Look for an optimized built-in or plugin implementation registered for the
   exact `type_uri`.
3. Allow that implementation to inspect the full `QuantizationProto`,
   standardized descriptors, and test vector before accepting the tensor.
4. Otherwise invoke the embedded decoder when present.
5. Reject the model if neither an accepted implementation nor an executable
   decoder is available.

For portable declarations, optimized implementations conform to the decoder
under normal ONNX numerical accuracy requirements. For opaque declarations,
implementations conform to the URI owner's external specification.
Bit-identical floating-point results are not required unless the format
specification independently requires them.

### Optional runtime plugins

Runtime-specific plugin APIs are permitted but are outside the ONNX
specification. A plugin can provide a decoder and fused kernels without changing
the model or runtime source:

```cpp
runtime.RegisterQuantizationPlugin(
    "org.tencent.sherry:stq1_0/v1",
    {
        .supports = [](const QuantizationProto& type) {
            // Inspect descriptors and optionally run the test vector.
            return supports_stq1_0(type);
        },
        .decode = decode_stq1_0,
        .matmul = matmul_stq1_0,
    });
```

For a portable declaration, the embedded function remains the semantics and
fallback. For an opaque declaration, the plugin or built-in codec defines the
implementation associated with the URI. A runtime MUST allow a plugin to
decline a declaration even when the URI matches.

## Checker Behavior and Forward Compatibility

The checker does not need built-in knowledge of every URI. It validates:

- URI presence and uniqueness within the model.
- Embedded decoder ABI, operator allowlist, and absence of required attributes
  when a decoder is present.
- `CUSTOM_QUANT` storage-field restrictions.
- Tensor/TypeProto annotation consistency.
- Prohibition of `CUSTOM_QUANT` in unsupported container and sparse types.
- External-data bounds.
- Standard descriptor names, types, and regular block bounds when present.
- Decoder function well-formedness under existing FunctionProto rules.

An unrecognized URI is not an error when its declaration and decoder are
present. An opaque declaration without a recognized codec is valid as a model
format but cannot be executed by that runtime. Tools that do not execute the
model can inspect, copy, rename, and round-trip either kind without
understanding the format.

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

The URI is a nominal contract, similar to a custom operator's domain and name.
Producers MUST ensure an embedded decoder matches the URI owner's definition.
For `onnx:` URIs, the checker MAY compare the declaration with the registered
standard definition. For other namespaces, runtimes and plugins decide whether
to accept the declaration.

This versioning is independent of the ONNX IR version. The IR version changes
only when the extensibility framework itself changes.

## Examples

The examples show declarations conceptually; function bodies are abbreviated.

### INT4 symmetric block quantization

```text
type_uri: "example.org:int4-symmetric-block32/v1"
decoder: FunctionProto<DecodeInt4Block32>
descriptor:
  values_per_block: 32
  bytes_per_block: 18
  field_names: ["values", "scale"]
  field_layout: [
    [0,   4, 32, 4,  UNDEFINED],
    [128, 16, 1, 16, FLOAT16],
  ]
```

The decoder extracts signed 4-bit values, bitcasts the embedded FP16 scale, and
multiplies the values by that scale.

### NF4

```text
type_uri: "example.org:nf4-block64/v1"
decoder: FunctionProto<DecodeNF4Block64>
descriptor:
  values_per_block: 64
  bytes_per_block: 34
  field_names: ["indices", "scale"]
  field_layout: [
    [0,   4, 64, 4,  UNDEFINED],
    [256, 16, 1, 16, FLOAT16],
  ]
```

The decoder contains the 16-entry NF4 codebook as a Constant, gathers values by
the unpacked indices, and applies the embedded scale.

### Sherry STQ1_0

STQ1_0 demonstrates why the decoder, not a closed family enum, must be
authoritative. One block stores multiple bit streams, an embedded scale, and a
non-contiguous logical element mapping.

```text
type_uri: "org.tencent.sherry:stq1_0/v1"
decoder: FunctionProto<DecodeSTQ1_0>
descriptor:
  values_per_block: 256
  bytes_per_block: 42
  field_names: ["code", "sign", "scale"]
  field_layout: [
    [0,   4, 64, 4,  UNDEFINED],
    [256, 1, 64, 1,  BOOL],
    [320, 16, 1, 16, FLOAT16],
  ]
```

For group `g` in `[0, 64)`, the decoder:

```text
code = extract_4_bits(payload, 4 * g)
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

### Opaque plugin-only format

Some proprietary formats may intentionally omit the decoder:

```text
type_uri: "vendor.example:accelerator-packed/v3"
decoder: <absent>
descriptor:
  vendor.example.kernel_abi: "matmul-v7"
test_payload: TensorProto<UINT8>
test_output: TensorProto<FLOAT>
```

The model remains structurally representable and round-trippable, but execution
requires a built-in implementation or plugin that accepts this declaration.
This is less portable than the previous examples and MUST NOT use an `onnx:`
URI.

## Backward Compatibility

- Models without extensible quantized tensors are unaffected.
- QDQ and built-in low-precision types remain unchanged.
- A new IR version gates the initial framework.
- Old runtimes reject models requiring the new IR version.
- New formats using the established framework do not require later IR bumps.

## Impact Assessment

### ONNX specification changes

- One opaque `CUSTOM_QUANT` data-type value.
- One new `QuantizationProto` message.
- New fields on ModelProto, TensorProto, and TypeProto.Tensor.
- A `DequantizeExtensible` operator.
- One new IR version to gate the framework.

### Runtime implementer burden

Minimum support consists of:

- Parsing and preserving the new fields.
- Validating payload and declaration structure.
- Invoking the embedded decoder for `DequantizeExtensible`.

Optimized support consists of dispatching built-in or plugin-provided fused
kernels by URI after accepting the declaration.

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

This RFC instead uses open-ended `AttributeProto` descriptors plus an
authoritative decoder. Convenience libraries may provide reusable decoder
functions for common families without placing those families in the IR.

### Store a decoder SHA-256 in the IR

A stored decoder digest does not add decoding semantics: the decoder is already
inside `QuantizationProto`, and a runtime can hash it locally if its plugin ABI
needs a cache or implementation key. Standardizing the digest would also
require canonicalization rules, would distinguish semantically equivalent
functions, and would become more complicated if decoder composition is allowed
later.

This RFC therefore does not serialize a decoder hash. Optimized implementations
match the URI and inspect the declaration before accepting it.

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

1. Should the standardized regular-layout descriptors be required for
   ONNX-governed URIs even though they remain optional for other formats?
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
