```toml
name = "NonZero"
domain = "ai.onnx"

[[inputs]]
name = "X"
type = "T"
desc = "input"
differentiation = "non_differentiable"

[[outputs]]
name = "Y"
type = "tensor(int64)"
desc = "output"
differentiation = "non_differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain to all tensor types."
```
