```toml
name = "Compress"
domain = "ai.onnx"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to all tensor types."

[type_constraints.T1]
types = ["tensor(bool)"]
desc = "Constrain to boolean tensors."
```
