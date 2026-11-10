```toml
name = "Expand"
domain = "ai.onnx"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor"
differentiation = "differentiable"

[[inputs]]
name = "shape"
type = "tensor(int64)"
desc = "A 1-D tensor indicates the shape you want to expand to, following the broadcast rule"
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Output tensor"
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to all tensors."
```
