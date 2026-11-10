```toml
name = "Sign"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor"
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "The sign of the input tensor computed element-wise. It has the same shape and type of the input."
differentiation = "non_differentiable"

[type_constraints.T]
types = "all_numeric_types_ir4"
desc = "Constrain input and output types to all numeric tensors."
```
