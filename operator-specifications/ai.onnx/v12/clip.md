```toml
name = "Clip"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[type_constraints.T]
types = "all_numeric_types"
desc = "Constrain input and output types to all numeric tensors."
```
