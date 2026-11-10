```toml
name = "Identity"
domain = "ai.onnx"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor"
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Tensor to copy input into."
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to all tensor types."
```
