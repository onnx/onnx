```toml
name = "TopK"
domain = "ai.onnx"
shape_inference = "defs::math::utils::TopKShapeInference"

[type_constraints.T]
types = "all_numeric_types_ir4"
desc = "Constrain input and output types to numeric tensors."

[type_constraints.I]
types = ["tensor(int64)"]
desc = "Constrain index tensor to int64"
```
