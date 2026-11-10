```toml
name = "GridSample"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::gridSampleShapeInference"

[type_constraints.T1]
types = "all_tensor_types_ir4"
desc = "Constrain input `X` and output `Y` types to all tensor types."

[type_constraints.T2]
types = "all_float_types_ir4"
desc = "Constrain grid types to float tensors."
```
