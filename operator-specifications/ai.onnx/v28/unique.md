```toml
name = "Unique"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::UniqueShapeInference_opset28"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Input can be of any tensor type."
```
