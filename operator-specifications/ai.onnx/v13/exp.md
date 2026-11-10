```toml
name = "Exp"
domain = "ai.onnx"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor"
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "The exponential of the input tensor computed element-wise"
differentiation = "differentiable"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."
```
