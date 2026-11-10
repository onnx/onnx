```toml
name = "ScatterND"
domain = "ai.onnx"

[[inputs]]
name = "data"
type = "T"
desc = "Tensor of rank r >= 1."
differentiation = "differentiable"

[[inputs]]
name = "indices"
type = "tensor(int64)"
desc = "Tensor of rank q >= 1."
differentiation = "non_differentiable"

[[inputs]]
name = "updates"
type = "T"
desc = "Tensor of rank q + r - indices_shape[-1] - 1."
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Tensor of rank r >= 1."
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to any tensor type."
```
