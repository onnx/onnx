```toml
name = "Constant"
domain = "ai.onnx"
shape_inference = "defs::generator::utils::ConstantShapeInference_opset9"

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to all tensor types."
```
