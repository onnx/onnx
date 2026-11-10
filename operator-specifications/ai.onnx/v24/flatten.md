```toml
name = "Flatten"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::FlattenShapeInference_opset24"

[type_constraints.T]
types = "all_tensor_types_ir12"
desc = "Constrain input and output to all tensor types up to IRv12."
```
