```toml
name = "Unsqueeze"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::UnsqueezeShapeInference_opset24"
data_propagation = "defs::tensor::utils::UnsqueezeDataPropagation_opset24"

[type_constraints.T]
types = "all_tensor_types_ir12"
desc = "Constrain input and output types to all tensor types up to IRv12."
```
