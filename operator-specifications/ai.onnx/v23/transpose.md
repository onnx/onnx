```toml
name = "Transpose"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::TransposeShapeInference_opset23"

[type_constraints.T]
types = "all_tensor_types_ir11"
desc = "Constrain input and output types to all tensor types."
```
