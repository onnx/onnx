```toml
name = "Det"
domain = "ai.onnx"
shape_inference = "defs::math::utils::DetShapeInference_opset22"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to floating-point tensors."
```
