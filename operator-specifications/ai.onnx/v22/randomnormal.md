```toml
name = "RandomNormal"
domain = "ai.onnx"
shape_inference = "defs::generator::utils::RandomNormalShapeInference_opset22"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain output types to float tensors."
```
