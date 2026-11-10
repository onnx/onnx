```toml
name = "Size"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::SizeShapeInference_opset19"
data_propagation = "defs::tensor::utils::SizeDataPropagation_opset19"

[type_constraints.T]
types = "all_tensor_types_ir9"
desc = "Input tensor can be of arbitrary type."

[type_constraints.T1]
types = ["tensor(int64)"]
desc = "Constrain output to int64 tensor, which should be a scalar though."
```
