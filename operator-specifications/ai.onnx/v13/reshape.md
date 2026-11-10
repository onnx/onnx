```toml
name = "Reshape"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::ReshapeShapeInference_opset13"

[[inputs]]
name = "data"
type = "T"
desc = "An input tensor."
differentiation = "differentiable"

[[inputs]]
name = "shape"
type = "tensor(int64)"
desc = "Specified shape for output."
differentiation = "non_differentiable"

[[outputs]]
name = "reshaped"
type = "T"
desc = "Reshaped data."
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to all tensor types."
```
