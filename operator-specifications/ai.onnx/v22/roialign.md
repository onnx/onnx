```toml
name = "RoiAlign"
domain = "ai.onnx"
shape_inference = "defs::object_detection::utils::RoiAlignShapeInference_opset22"

[type_constraints.T1]
types = "all_float_types_ir4"
desc = "Constrain types to float tensors."

[type_constraints.T2]
types = ["tensor(int64)"]
desc = "Constrain types to int tensors."
```
