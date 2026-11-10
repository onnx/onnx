```toml
name = "Squeeze"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::SqueezeShapeInference_opset21"
data_propagation = "defs::tensor::utils::SqueezeDataPropagation_opset21"

[type_constraints.T]
types = "all_tensor_types_ir10"
desc = "Constrain input and output types to all tensor types up to IRv10."
```
