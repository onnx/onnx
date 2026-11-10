```toml
name = "Shape"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::ShapeShapeInference_opset13"
data_propagation = "defs::tensor::utils::ShapeOp13DataPropagator"

[[inputs]]
name = "data"
type = "T"
desc = "An input tensor."
differentiation = "non_differentiable"

[[outputs]]
name = "shape"
type = "T1"
desc = "Shape of the input tensor"
differentiation = "non_differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Input tensor can be of arbitrary type."

[type_constraints.T1]
types = ["tensor(int64)"]
desc = "Constrain output to int64 tensor."
```
