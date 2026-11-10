```toml
name = "Log"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor"
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "The natural log of the input tensor computed element-wise"
differentiation = "differentiable"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."
```
