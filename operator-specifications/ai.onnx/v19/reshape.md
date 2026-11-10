```toml
name = "Reshape"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::ReshapeShapeInference_opset19"

[type_constraints.T]
types = "all_tensor_types_ir9"
desc = "Constrain input and output types to all tensor types."
```
