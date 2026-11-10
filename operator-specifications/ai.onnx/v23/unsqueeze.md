```toml
name = "Unsqueeze"
domain = "ai.onnx"

[[inputs]]
name = "data"
type = "T"
desc = "Original tensor"
differentiation = "differentiable"

[[inputs]]
name = "axes"
type = "tensor(int64)"
desc = "1D tensor of integers indicating the dimensions to be inserted. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(expanded)."
differentiation = "non_differentiable"

[type_constraints.T]
types = "all_tensor_types_ir11"
desc = "Constrain input and output types to all tensor types up to IRv11."
```
