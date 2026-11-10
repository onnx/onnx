```toml
name = "Sum"
domain = "ai.onnx"

[[inputs]]
name = "data_0"
type = "T"
desc = "List of tensors for sum."
option = "variadic"
differentiation = "differentiable"

[[outputs]]
name = "sum"
type = "T"
desc = "Output tensor."
differentiation = "differentiable"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."
```
