# RFC: Extensible Quantization Type System for ONNX

| | |
|---|---|
| **Authors** | Justin Chu (@justinchuby) |
| **Status** | Draft — design discussion |
| **Created** | 2025-07-22 |
| **ONNX Issue** | TBD |
| **Spec Impact** | ModelProto, GraphProto, TypeProto, new operator support, new IR version |

## Abstract

This RFC explores an open-world quantization type system for ONNX. A model
should be able to describe new quantized formats without adding a new built-in
data type or incrementing the IR version for every format.

The current candidate design represents one logical quantized tensor as a
composite of one or more typed storage tensors. This matches existing model
formats:

- GPTQ, AWQ, ONNX Runtime MatMulNBits, MXFP4/NVFP4, AQLM, and SpQR commonly
  store packed values, scales, zero points, codebooks, or sparse metadata as
  separate tensors.
- GGUF/ggml formats, including Sherry STQ1_0, store one self-contained stream
  of quantized blocks.

Each quantization type embeds a whole-tensor ONNX decoder. Runtime plugins are
optional accelerators and do not define otherwise-missing model semantics.

This document records the preferred direction and the remaining design
questions. The protobuf sketches are illustrative, not final.

## Requirements

### Functional requirements

1. New quantization formats do not require new ONNX element types.
2. Logical tensor shape is independent of physical storage shape and size.
3. A format may use one opaque byte stream or multiple typed storage tensors.
4. Scales, zero points, codebooks, sparse metadata, and permutations can remain
   separate and share storage where appropriate.
5. Every declaration contains a deterministic, self-contained ONNX decoder.
6. Runtime plugins can provide decoding and fused kernels without changing the
   model.
7. Existing QDQ models and built-in low-precision types remain unchanged.
8. Unknown declarations and storage can be preserved and round-tripped.

### Design requirements

- The IR should add as few new message kinds as practical.
- Quantization-family enums or closed `oneof` taxonomies must not become the
  extension boundary.
- Physical component dtypes remain explicit. A scale tensor remains FLOAT16;
  a packed word tensor remains UINT8 or INT32.
- Storage tensors are not ordinary graph values and cannot accidentally be
  consumed by normal nodes.
- URI versioning is independent of the ONNX IR version.

### Non-goals for the first revision

- Defining an encoder or `QuantizeExtensible`.
- Defining activation-quantization policy annotations.
- Guaranteeing optimized execution for every runtime.
- Standardizing a runtime plugin ABI.
- Supporting every container or control-flow use of a quantized value.

## Evidence from Existing Formats

| Format | Storage components for one logical weight | Layout |
|--------|-------------------------------------------|--------|
| GPTQ | `qweight`, `qzeros`, `scales`, often `g_idx` | Usually grouped along the input/K axis |
| AWQ | `qweight`, `qzeros`, `scales`; kernel-specific variants | K-axis groups or prepacked tiles |
| ORT MatMulNBits | packed `B`, `scales`, optional `zero_points` | K-axis blocks, separate operator inputs |
| MXFP4 | packed blocks and E8M0 scales are often separate tensors | Blocks of 32 values |
| NVFP4 | packed values, per-block scales, tensor scale | Blocks of 16 values, optional backend swizzles |
| AQLM | `codes`, shared `codebooks`, `scales` | Multidimensional group grid |
| SpQR | dense packed core, sparse offsets/values, optional permutation | Tiled dense data plus variable sparse data |
| GGUF K-quants | One byte stream with embedded scales/mins | Fixed-size row blocks |
| Sherry STQ1_0 | One byte stream with codes, sign bits, FP16 scale | 42 bytes per 256 logical values |

A mandatory single payload is lossless in principle but forces copying or
repacking for most Hugging Face and ONNX Runtime representations. It can also
destroy useful sharing, such as an AQLM codebook referenced by multiple logical
tensors.

The candidate design therefore treats one logical quantized value as a
composite over typed storage tensors. A one-component UINT8 payload remains a
fully supported fast path.

## Agreed Direction

The following points were agreed during design discussion:

1. Every quantization declaration includes a decoder. Plugins accelerate the
   declaration; they are not the sole source of semantics.
2. The decoder is embedded in the quantization declaration and is not added to
   `ModelProto.functions`.
3. QuantizedTensorProto and TypeProto shapes always mean decoded logical shape.
   Component and parameter shapes describe their physical storage.
4. Declarations are model-global and referenced directly by URI.
5. The decoder operates on the whole logical tensor, not one block at a time.
6. No encoder or `QuantizeExtensible` is introduced in this RFC.
7. Decoder SHA fields and embedded test vectors are not required.
8. `DequantizeExtensible` has an `output_dtype` attribute.
9. If quantization-preserving operations are introduced, they rely on
   runtime/plugin capability rather than per-operator functions embedded in the
   declaration.
10. Fixed-block metadata is optional. Formats without it remain decodable but
    lose generic block-size validation and related optimizations.

## Candidate Data Model

### QuantizationParameterProto

Parameters are compile-time refinements of a quantization type, such as group
size, bit width, or block axis. Runtime-varying arrays such as scales and
codebooks are storage components, not parameters.

```protobuf
message QuantizationParameterProto {
  optional string name = 1;

  // Restricted to an ordinary dense, non-quantized tensor type.
  optional TypeProto type = 2;

  // Missing means the parameter is required.
  // Defaults are small, concrete, inline tensor literals.
  optional TensorProto default_value = 3;

  optional string doc_string = 4;
}
```

Instance bindings use named `TensorProto` literals. Their `name` fields identify
the parameters being bound. External data and symbolic dimensions are not
allowed for parameter values.

Whether component bindings and parameter bindings should instead be unified is
an open question.

### QuantizationProto

```protobuf
message QuantizationProto {
  // Immutable, globally unique format identity including semantic version.
  optional string uri = 1;

  // Ordered storage-component roles and their physical types. This is the
  // normative component interface.
  repeated ValueInfoProto component = 2;

  repeated QuantizationParameterProto parameter = 3;

  // Whole-tensor portable decoder.
  optional FunctionProto decoder = 4;

  // The logical element type produced by the decoder.
  optional int32 logical_elem_type = 5;  // TensorProto.DataType

  optional string doc_string = 6;
  repeated StringStringEntryProto metadata_props = 7;
}
```

Component roles are format-defined. Examples include `payload`, `qweight`,
`scales`, `zero_points`, `codebooks`, `row_offsets`, and `outliers`.
Each component `ValueInfoProto.type` is restricted to an ordinary dense,
non-quantized tensor type.

The URI identifies the component interface, parameter schema, decoder
semantics, and logical element type. A semantic change requires a new URI.

The decoder's component inputs and type annotations must agree with this
normative component interface. They do not define a second component schema.

### QuantizedTensorProto

```protobuf
message QuantizedTensorProto {
  optional string name = 1;
  repeated int64 dims = 2;  // Logical decoded shape.
  optional string quantization_uri = 3;

  // Component role -> quantization-storage tensor name.
  repeated StringStringEntryProto component = 4;

  // TensorProto.name is the parameter name.
  repeated TensorProto parameter = 5;

  optional string doc_string = 6;
  repeated StringStringEntryProto metadata_props = 7;
}
```

Each component role must be bound exactly once unless the declaration defines a
default for that role in a future revision. Component storage may be shared by
multiple logical quantized tensors.

### Storage pool and quantized initializers

The preferred placement after architecture review is model-global storage:

```protobuf
message ModelProto {
  // ... existing fields ...
  repeated QuantizationProto quantization = TBD;
  repeated TensorProto quantization_storage = TBD;
}

message GraphProto {
  // ... existing fields ...
  repeated QuantizedTensorProto quantized_initializer = TBD;
}
```

`quantization_storage` is not part of the graph value namespace:

- Every storage tensor has a globally unique name.
- Storage names are disjoint from all graph value names.
- A `NodeProto` input cannot reference storage directly.
- Multiple quantized initializers may share storage.
- Unreferenced storage is legal; removal is a separate garbage-collection pass.
- Each storage tensor retains normal TensorProto dtype, shape, raw data, and
  external-data behavior.

`QuantizedTensorProto.name` participates in the normal graph value namespace and
must be unique across dense, sparse, and quantized initializers. If v1 is
restricted to quantized initializers, a quantized initializer cannot also
appear in `graph.input` as an overridable default.

Model-global placement avoids outer-graph capture rules, cross-subgraph
duplication, and ambiguous ownership. Graph-local storage remains an
alternative under consideration.

### TypeProto.QuantizedTensor

If composite quantized values are represented in the graph type system, the
candidate type is:

```protobuf
message TypeProto {
  message QuantizedTensor {
    optional string quantization_uri = 1;
    optional int32 elem_type = 2;  // Logical element type.
    optional TensorShapeProto shape = 3;
  }

  oneof value {
    // ... existing fields ...
    QuantizedTensor quantized_tensor_type = TBD;
  }
}
```

A new `oneof` arm fails closed in existing tooling. This is safer than adding
annotations to `TypeProto.Tensor`, which could cause ordinary tensor operators
to accept packed storage accidentally.

Whether this type arm is required in the first revision is a major open
question. A smaller variadic-operator alternative is described below.

Parameter values are intentionally not stored in `TypeProto`. They belong to
the concrete quantized value; putting TensorProto data inside a type would make
type equality, printing, and shape merging depend on arbitrary serialized
values.

## Decoder Contract

The candidate decoder ABI is:

```text
Decode(
    logical_shape: tensor(int64)[rank],
    component_0,
    ...,
    component_N,
    parameter_0,
    ...,
    parameter_M
) -> values: tensor(logical_elem_type)[logical_shape]
```

Rules:

- The first input is reserved for the logical shape.
- Component inputs follow `QuantizationProto.component` order.
- Parameter inputs follow `QuantizationProto.parameter` order after defaults
  have been applied.
- Every input and output has mandatory `ValueInfoProto` type information.
- Decoder component inputs and value information agree with the declaration's
  normative component list.
- The output type matches `logical_elem_type`.
- The output shape at invocation matches the logical shape.
- The decoder has no captures and does not reference model-local functions or
  custom-domain operators.
- The decoder uses only deterministic, self-contained standard ONNX operators
  from its own `opset_import`.
- Decoder constants are inline. They do not use external data.

The embedded FunctionProto is scoped only to its QuantizationProto. Its
`name`, `domain`, and `overload` do not register a callable model-local
function. Its `opset_import` controls resolution of nodes in the decoder body
independently of `ModelProto.functions`.

The exact mechanically checkable restriction on decoder operators remains an
open question. A likely rule is to reject operators with `GRAPH`/`GRAPHS`
attributes and known random or nondeterministic operators, rather than maintain
a closed allowlist.

## Runtime Value and Loading

A runtime quantized value consists of:

```text
{
  declaration,
  logical_shape,
  effective_static_parameters,
  named_component_tensor_handles
}
```

### Model load

1. Parse declarations and require unique URIs.
2. Parse and bounds-check model-global storage tensors.
3. Resolve each quantized initializer's URI.
4. Apply parameter defaults and validate explicit bindings.
5. Require component bindings to match declaration roles exactly.
6. Validate component dtypes and ranks against declaration `ValueInfoProto`s.
7. Construct immutable component handles without concatenating their buffers.
8. Ask execution providers/plugins whether they support required operations for
   the URI, parameters, component shapes/dtypes, and requested output dtype.

The URI is a nominal dispatch key, similar to a custom-op domain and name.
Plugins must inspect the declaration interface and effective parameters before
accepting it. A serialized decoder digest is not necessary.

### External data and zero-copy

Each component independently uses existing `TensorProto.external_data`.

- Explicit offsets and lengths are required for external quantization storage.
- Shared or overlapping external ranges are allowed.
- Zero-copy loading is possible when the native format matches the component
  TensorProto dtype and ONNX packing.
- Formats with different nibble ordering or alignment use UINT8 components
  rather than pretending to be ONNX INT4/INT2.
- Tools that save, load, extract, or compose models must traverse
  `quantization_storage` and preserve sharing.

## DequantizeExtensible

```text
DequantizeExtensible(X: quantized_tensor(T)) -> Y

attribute:
  output_dtype: TensorProto.DataType = UNDEFINED
```

Semantics:

1. Invoke the declaration's decoder to obtain `tensor(logical_elem_type)`.
2. If `output_dtype` differs, apply the semantics of `Cast`.

The output shape equals the logical input shape. When `output_dtype` is
UNDEFINED, type inference resolves it to the declaration's
`logical_elem_type`. An explicit value applies Cast semantics.

Execution providers may fuse decoding, output conversion, and consuming
operators without materializing the decoded tensor.

## Standard Operators and MoE Gather

The initial design goal was a quantization-preserving Gather:

```text
Gather(quantized_tensor, indices) -> quantized_tensor
```

Architecture review found this circular without an encoder or a format-specific
storage transform. A runtime would have to create new packed components whose
canonical representation is not defined by the decoder.

The recommended portable alternative is:

```text
Gather(quantized_tensor(T), indices) -> tensor(T)
```

with semantics:

```text
Gather(Decode(X), indices)
```

This requires no encoder. A plugin can gather only the selected component
ranges and decode them, or fuse Gather with the following MatMul. The reference
semantics may decode more data but remain correct.

This pattern covers MoE selection when optimized by a capable runtime, but the
graph no longer carries a quantized value across the Gather boundary.

Whether this is sufficient for MoE, or whether quantization-preserving Gather
must be deferred until storage-transform semantics exist, is an open question.

Other operators may adopt the same rule in later opsets: accepting a quantized
logical input is equivalent to applying the operator to its decoded value.

## Plugin Capability Negotiation

The plugin API is runtime-specific and outside the ONNX specification. A
conceptual query is:

```text
supports(
    quantization_uri,
    declaration,
    effective_parameters,
    component_types_and_shapes,
    operation,
    output_dtype,
    device
)
```

Plugins may provide:

- dequantization;
- fused MatMul/Gemm;
- decoded-output Gather for embeddings or MoE;
- serialization-preserving operations outside standard ONNX semantics.

The model remains semantically complete without a plugin.

## Checker Requirements

The checker should validate:

- declaration URIs are present and unique;
- every declaration has a decoder;
- decoder inputs, component roles, parameters, and output type are consistent;
- parameter defaults and bindings are concrete and type-correct;
- storage names are globally unique and disjoint from graph value names;
- quantized initializer names are unique in the graph value namespace;
- quantized initializers do not appear in `graph.input` when graph input support
  is disabled;
- every component binding resolves to model storage;
- component dtypes and ranks satisfy declaration constraints;
- quantized initializer logical shape and URI agree with corresponding type
  information;
- external-data offsets and lengths are valid;
- decoder functions are deterministic and self-contained under the final
  structural rule.
- quantized values are not used in Sequence, Optional, Map, Loop/Scan carried
  values, or graph boundaries when those v1 restrictions are selected.

Model-composition tooling must merge structurally identical declarations with
the same URI into one declaration. Conflicting declarations for one URI are a
hard error; a valid resulting model still contains exactly one declaration per
URI.

The checker cannot generally prove that arbitrary component shapes encode the
declared logical shape. An optional fixed-block profile can add stronger checks.

## Optional Fixed-Block Profile

Many common formats can expose reserved static parameters:

```text
__block_axis
__values_per_block
__bytes_per_block
```

The simple profile above is sufficient only for a declaration with one
component named `payload`. When all are present, the checker can validate that
stream and runtimes can recognize a common block layout. Multi-component
formats need component-specific size rules that have not yet been designed.
Formats with multidimensional tiles, sparse outliers, or variable-length
components omit the profile and rely on their decoder.

Whether the first revision should standardize only a single-axis profile or
also support multidimensional tile shapes remains open.

## Examples

### Sherry STQ1_0

Declaration:

```text
uri: "urn:org.tencent.sherry:stq1_0:v1"
logical_elem_type: FLOAT
components:
  payload: tensor(uint8)[...]
parameters:
  __block_axis: int64
  __values_per_block: int64 = 256
  __bytes_per_block: int64 = 42
```

One block contains:

```text
code:  64 x 4-bit values = 32 bytes
sign:  64 x 1-bit values = 8 bytes
scale: 1 x FLOAT16       = 2 bytes
```

For group `g`:

```text
code = extract_4_bits(payload, 4 * g)
sign = extract_1_bit(payload, 256 + g)
vector = codebook[16 * sign + code]

chunk = g / 16
lane = g % 16
output_index(p) = 64 * chunk + lane + 16 * p
output[output_index(p)] = vector[p] * scale
```

This format uses one UINT8 storage component.
The fixed 32 x 4 ternary codebook is an inline Constant in the decoder.

### GPTQ

```text
uri: "urn:example:gptq:v1"
components:
  qweight: tensor(int32)[...]
  qzeros: tensor(int32)[...]
  scales: tensor(float16)[...]
  g_idx: tensor(int32)[...]
parameters:
  bits: int64
  group_size: int64
```

The decoder reconstructs the logical weight without concatenating components.

### AQLM

```text
uri: "urn:example:aqlm:v1"
components:
  codes: tensor(int32)[...]
  codebooks: tensor(float16)[...]
  scales: tensor(float16)[...]
```

Multiple quantized tensors may reference the same `codebooks` storage tensor.

## Smaller Alternative: Variadic DequantizeExtensible

A substantially smaller design is:

```text
DequantizeExtensible(
    component_0,
    ...,
    component_N
) -> tensor(output_dtype)

attributes:
  quantization_uri
  logical_shape
  parameters
  output_dtype
```

All components remain ordinary graph initializers. This requires:

- one quantization declaration message;
- no `QuantizedTensorProto`;
- no new TypeProto arm;
- no storage namespace.

It reuses existing external-data, mmap, composition, and runtime APIs.

The tradeoff is that the logical quantized value is no longer self-describing:
its identity exists only at the consuming node, intermediate quantized values
cannot be typed, and tools must recover the relationship from node inputs.

If self-describing quantized values are a hard requirement, the composite type
design is preferable. Otherwise this alternative has much lower ecosystem cost
and should be explicitly evaluated before implementation.

## Alternatives Considered

### Single opaque payload with CUSTOM_QUANT

This minimizes IR changes and works naturally for GGUF/STQ. It forces copying,
repacking, or loss of sharing for common multi-component formats and does not
solve graph-input buffer sizing cleanly.

### Closed quantization-family protobuf hierarchy

A closed hierarchy is readable for known formats but makes protobuf cases the
extension boundary. New combinations of storage and decoding semantics would
require IR changes.

### Decoder SHA-256

A stored digest adds no semantics and requires canonicalization rules. Runtimes
may compute a local digest for caching, while plugin acceptance remains nominal
on URI plus structural interface checks.

### Embedded test vectors

The decoder is the semantic oracle. Test suites can be maintained by the URI
owner without adding duplicate expected outputs to every model.

## Open Questions and Recommendations

### 1. Storage pool scope

**Recommendation:** model-global `ModelProto.quantization_storage`.

Graph-local storage complicates subgraph capture, cross-graph sharing,
extraction, and ownership.

### 2. Separate component and parameter mechanisms

**Current candidate:** `ValueInfoProto` component roles plus
`QuantizationParameterProto` static parameters.

**Reviewer alternative:** one ordered formal parameter list with optional
defaults, covering payloads, scales, codebooks, and static parameters.

**Recommendation:** keep the semantic distinction because static parameters are
part of type compatibility while components are runtime storage, but prototype
both checker paths before finalizing.

### 3. Logical element type

**Recommendation:** declare `logical_elem_type` on `QuantizationProto` and
require the decoder output to match it.

Always decoding to FLOAT can introduce double-rounding differences for formats
whose reference semantics are FLOAT16 or BFLOAT16.

### 4. DequantizeExtensible default output type

**Recommendation:** give the schema attribute the static default UNDEFINED.
During type inference, UNDEFINED resolves to `logical_elem_type`, following the
pattern used by DequantizeLinear. An explicit `output_dtype` has Cast semantics.

### 5. TypeProto.QuantizedTensor in the first revision

**Composite-design recommendation:** include it because self-describing logical
quantized values are a stated requirement.

**Reviewer concern:** a new type arm affects every checker, parser, printer,
shape-inference, composition, and runtime API path. SparseTensor demonstrates
the risk of landing a type with too few consuming operators.

**Alternative:** first ship variadic `DequantizeExtensible`, then add the type
arm with the second consuming operator.

### 6. Graph inputs, outputs, and control-flow boundaries

**Recommendation for v1:** allow quantized initializers only. Do not permit
quantized graph inputs/outputs, Sequence/Optional/Map elements, or Loop/Scan
carried values until a public composite runtime ABI is defined.

FunctionProto has no initializer field, so quantized initializers also cannot
exist directly inside a model-local function. Extracting a subgraph containing
one into a FunctionProto would require lifting the quantized value and its
storage to model scope or adding a later function-storage mechanism.

### 7. Gather semantics

**Recommendation:** if Gather support is included, make its output an ordinary
decoded tensor with semantics `Gather(Decode(X), indices)`. Plugins may fuse
selection and decoding.

Quantization-preserving Gather is not well-defined without an encoder or
format-specific storage-transform semantics.

### 8. Fixed-block profile

**Recommendation:** optional single-axis profile in v1, normatively validated
when present. Multi-axis tiles remain representable through the decoder but do
not receive generic block validation.

### 9. Decoder restrictions

**Recommendation:** standard-domain operators only, no custom/model-local
calls, no graph-valued attributes, no known random/nondeterministic operators.

A fixed allowlist is too rigid; an unrestricted "deterministic" requirement is
not mechanically checkable.

### 10. URI governance

**Recommendation:** opaque absolute ASCII identifiers, preferably URNs or
reverse-DNS namespaces. Lookup performs exact byte comparison with no
normalization.

The URI covers component ABI, parameter schema, decoder, and logical element
type. Merging models with conflicting declarations for one URI is an error.

### 11. Plugin acceptance without digest or test vectors

**Recommendation:** dispatch nominally by URI, then require structural
compatibility of component roles/types, parameter schema, and logical element
type before a plugin accepts the declaration.

### 12. Implementation cost

The RFC should explicitly list required updates to:

- checker and type-constraint grammar;
- parser and printer;
- shape inference helpers;
- external-data traversal;
- model extraction and composition;
- function extraction/inlining rules for quantized initializers;
- version conversion;
- runtime public value APIs if graph I/O is later enabled.

## Backward Compatibility and Versioning

- Existing models are unchanged.
- Existing QDQ and built-in data types remain valid.
- One IR-version bump introduces the framework.
- New quantization declarations and URIs do not require later IR-version bumps.
- New operator support may require opset versions without changing the IR.

## References

- [AutoGPTQ packed linear storage](https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_cuda_old.py)
- [AutoAWQ quantized linear layouts](https://github.com/casper-hansen/AutoAWQ/tree/main/awq/modules/linear)
- [ONNX Runtime MatMulNBits quantizer](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/matmul_nbits_quantizer.py)
- [OCP Microscaling Formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [AQLM inference representation](https://github.com/Vahe1994/AQLM/tree/main/inference_lib/src/aqlm)
- [SpQR inference representation](https://github.com/Vahe1994/SpQR/tree/main/inference_lib/src/spqr_quant)
- [llama.cpp quantization block definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)
- [Sherry STQ1_0 implementation PR](https://github.com/ggml-org/llama.cpp/pull/22836)
