```toml
name = "Flatten"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::FlattenShapeInference_opset25"

[type_constraints.T]
types = "all_tensor_types_ir13"
desc = "Constrain input and output to all tensor types up to IRv13."
```
