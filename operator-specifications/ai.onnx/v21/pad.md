```toml
name = "Pad"
domain = "ai.onnx"

[type_constraints.T]
types = "all_tensor_types_ir10"
desc = "Constrain input and output types to all tensor types up to IRv10."

[type_constraints.Tind]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain indices to integer types"
```
