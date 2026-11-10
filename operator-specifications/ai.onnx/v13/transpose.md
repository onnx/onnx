```toml
name = "Transpose"
domain = "ai.onnx"

[[inputs]]
name = "data"
type = "T"
desc = "An input tensor."
differentiation = "differentiable"

[[outputs]]
name = "transposed"
type = "T"
desc = "Transposed output."
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to all tensor types."
```
