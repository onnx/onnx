```toml
name = "Shape"
domain = "ai.onnx"

[type_constraints.T]
types = "all_tensor_types_ir11"
desc = "Input tensor can be of arbitrary type."

[type_constraints.T1]
types = ["tensor(int64)"]
desc = "Constrain output to int64 tensor."
```
