<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Checker error codes

When `onnx.checker` rejects a model, the `ValidationError` it raises carries a `code`: a stable identifier of the rule that failed. Code that reacts to particular failures should compare codes, not messages. Messages may be reworded and include model-specific names; codes do not change between releases.

```python
import onnx

try:
    onnx.checker.check_model(model)
except onnx.checker.ValidationError as e:
    if e.code == "external-data-file-missing":
        ...
```

In C++, the code is `ValidationError::code()`, and the constants are in `onnx::checker::error_code`.

Not every rule has a code yet. Errors from rules without one have `code` set to `None` in Python and an empty `code()` in C++. Codes are added over time; once added, a code keeps its name and meaning.

| Code | Rule |
|---|---|
| `ir-version-missing` | The model does not set `ir_version`. |
| `ir-version-unsupported` | The model's `ir_version` is newer than this version of ONNX supports. |
| `metadata-props-duplicate-key` | Two `metadata_props` entries of the model have the same key. |
| `required-field-missing` | A field the specification requires is not set. |
| `required-field-empty` | A field the specification requires to be non-empty is empty, such as a graph or initializer name. |
| `data-type-requires-newer-ir-version` | A tensor or type uses a data type that was introduced after the model's `ir_version`. |
| `opset-import-missing` | A node uses a domain the model or function does not import, or a model with IR version 3 or later imports no opset. |
| `operator-not-registered` | No operator schema exists for a node in a standard domain at the imported opset version, or in any domain when `check_custom_domain=True`. |
| `operator-deprecated` | A node uses an operator that is deprecated at the imported opset version. |
| `node-schema-mismatch` | A node does not match its operator schema: it has too few or too many inputs or outputs, an empty name for an input or output that must be set, or an attribute that is duplicated, unknown, missing, or of the wrong attribute type. The types of the inputs and outputs are not part of this check. `onnx.shape_inference.infer_node_outputs` reports the same code. |
| `input-not-defined` | A node uses a value that no earlier node, graph input, initializer or outer scope defines: either only a later node produces it, so the nodes are not topologically sorted, or nothing does. |
| `not-ssa` | A value name is assigned more than once, for example by two nodes, two graph inputs or two initializers. |
| `graph-output-not-produced` | A graph output is not produced inside the graph. |
| `external-data-entry-invalid` | An externally stored tensor also has inline data, lacks a `location`, or has an `offset` or `length` that is not a non-negative integer. |
| `external-data-location-invalid` | An external data `location` is empty, absolute, outside the model directory, a symbolic link, a hard link, or not a regular file. |
| `external-data-file-missing` | The file an external data `location` names does not exist. |
| `external-data-out-of-range` | An external data `offset` or `length` lies beyond the end of the file, or the data is too small for the tensor's shape and type. |
