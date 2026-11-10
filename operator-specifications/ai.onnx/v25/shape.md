```toml
name = "Shape"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::ShapeShapeInference_opset25"
data_propagation = "defs::tensor::utils::ShapeDataPropagation_opset25"

[type_constraints.T]
types = "all_tensor_types_ir13"
desc = "Input tensor can be of arbitrary type."

[type_constraints.T1]
types = ["tensor(int64)"]
desc = "Constrain output to int64 tensor."
```
