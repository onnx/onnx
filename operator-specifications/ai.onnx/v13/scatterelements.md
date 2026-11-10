```toml
name = "ScatterElements"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::ScatterElementsShapeInference_opset13"

[[inputs]]
name = "data"
type = "T"
desc = "Tensor of rank r >= 1."
differentiation = "differentiable"

[[inputs]]
name = "indices"
type = "Tind"
desc = "Tensor of int32/int64 indices, of r >= 1 (same rank as input). All index values are expected to be within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds."
differentiation = "non_differentiable"

[[inputs]]
name = "updates"
type = "T"
desc = "Tensor of rank r >=1 (same rank and shape as indices)"
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Tensor of rank r >= 1 (same rank as input)."
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Input and output types can be of any tensor type."

[type_constraints.Tind]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain indices to integer types"
```
