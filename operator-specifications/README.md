<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->

# Operator specifications

This directory holds the declarative part of every ONNX operator schema:
doc string, attributes, inputs, outputs, type constraints, static function bodies and further settings,
as well as the names of the type and shape inference, data propagation
and context-dependent function body builder functions.
At build time, `tools/operator_specs` generates C++ functions from these files
that return the corresponding `OpSchema`.
The named functions themselves are hand-written C++ in `onnx/defs/<area>/utils.cc`.

The design is described in
[proposal 0009](../docs/proposals/0009-MarkdownSchemaGenerationProposal.md).

## Layout

```
<domain>/v<N>/<op>.md
```

The file exists if and only if operator `<op>` of `<domain>` has a schema with `since_version == N`.
`<op>` is the lowercase operator name.

## Format

A file is a fenced TOML block followed by the doc string in Markdown:

````markdown
```toml
name = "LeakyRelu"
domain = "ai.onnx"

[attributes.alpha]
type = "FLOAT"
desc = "Coefficient of leakage."
default = 0.01

[[inputs]]
name = "X"
type = "T"
desc = "Input tensor"
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor"
differentiation = "differentiable"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."
```

LeakyRelu takes input data (Tensor<T>) and an argument alpha, ...
````

| Key | Value |
|-----|-------|
| `name`, `domain` | Operator name and domain; required in every file and must match the path. |
| `support_level` | `"common"` (default) or `"experimental"`. |
| `deprecated` | `true` or `false` (default). |
| `node_determinism` | `"unknown"` (default), `"deterministic"` or `"non_deterministic"`. |
| `num_outputs` | List of the allowed numbers of outputs; by default, they follow from `outputs`. |
| `allow_unchecked_attributes` | `true` or `false` (default). |
| `shape_inference` | Name of the type and shape inference function, or `false` if the schema has none; required in every file. |
| `data_propagation` | Name of the data propagation function; optional. |
| `function_body_builders` | Map `opset_version -> name` of the context-dependent function body builders; optional. |
| `attributes` | Map `name -> {type, desc, required, default}`; `type` is an `AttributeProto` type such as `"INT"` or `"FLOATS"`. |
| `inputs`, `outputs` | Ordered list of `{name, type, desc, option, homogeneous, min_arity, differentiation}`; `option` is `"single"` (default), `"optional"` or `"variadic"`; `differentiation` is `"unknown"` (default), `"differentiable"` or `"non_differentiable"`. |
| `type_constraints` | Map `name -> {types, desc}`; `types` is a list of type strings such as `["tensor(float)"]` or the name of an `OpSchema` type helper such as `"all_float_types_ir4"`. |
| `function_bodies` | Map `opset_version -> ONNX text` of the function body. |

The common indentation of all doc string lines but the first is removed.

Function names are C++ names relative to `ONNX_NAMESPACE`.
A name `defs::<area>::utils::<name>` must be declared in `onnx/defs/<area>/utils.h`,
an unqualified name in `onnx/defs/shape_inference.h` or `onnx/defs/data_propagators.h`.

## Inheritance

Each file inherits from the previous version of the same operator:

- key absent: inherited unchanged;
- key present: replaces the inherited value completely (lists and maps are restated in full);
- doc string empty: inherited.

`name`, `domain`, `shape_inference`, `data_propagation` and `function_body_builders` are not inherited.
Every schema should have a type and shape inference function.

## Adding a version of an operator

1. Create `<domain>/v<N>/<op>.md` with `name`, `domain` and the keys that change.
2. Register the generated schema in `onnx/defs/<area>/defs.cc`
   and move the registration of the previous version to `onnx/defs/<area>/old.cc`:

   ```cpp
   ONNX_OPERATOR_SET_SCHEMA(
       LeakyRelu,
       16,
       generated::ai_onnx::LeakyRelu_ver16().TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));
   ```

3. Add the schema to the operator set in `onnx/defs/operator_sets*.h`.
