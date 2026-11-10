```toml
name = "Flatten"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::FlattenShapeInference_opset9"

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output to all tensor types."
```
