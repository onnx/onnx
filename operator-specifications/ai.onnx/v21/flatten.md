```toml
name = "Flatten"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::FlattenShapeInference_opset21"

[type_constraints.T]
types = "all_tensor_types_ir10"
desc = "Constrain input and output to all tensor types up to IRv10."
```
