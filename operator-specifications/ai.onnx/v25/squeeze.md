```toml
name = "Squeeze"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::SqueezeShapeInference_opset25"
data_propagation = "defs::tensor::utils::SqueezeDataPropagation_opset25"

[type_constraints.T]
types = "all_tensor_types_ir13"
desc = "Constrain input and output types to all tensor types up to IRv13."
```
