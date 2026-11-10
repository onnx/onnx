```toml
name = "Unsqueeze"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::UnsqueezeShapeInference_opset25"
data_propagation = "defs::tensor::utils::UnsqueezeDataPropagation_opset25"

[type_constraints.T]
types = "all_tensor_types_ir13"
desc = "Constrain input and output types to all tensor types up to IRv13."
```
