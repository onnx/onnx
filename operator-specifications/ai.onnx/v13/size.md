```toml
name = "Size"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::SizeShapeInference_opset13"
data_propagation = "defs::tensor::utils::SizeDataPropagation_opset13"

[[inputs]]
name = "data"
type = "T"
desc = "An input tensor."
differentiation = "non_differentiable"

[[outputs]]
name = "size"
type = "T1"
desc = "Total number of elements of the input tensor"
differentiation = "non_differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Input tensor can be of arbitrary type."

[type_constraints.T1]
types = ["tensor(int64)"]
desc = "Constrain output to int64 tensor, which should be a scalar though."
```
