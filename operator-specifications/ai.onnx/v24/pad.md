```toml
name = "Pad"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::PadShapeInference"

[type_constraints.T]
types = "all_tensor_types_ir12"
desc = "Constrain input and output types to all tensor types up to IRv12."

[type_constraints.Tind]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain indices to integer types"
```
