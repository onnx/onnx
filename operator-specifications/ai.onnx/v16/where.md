```toml
name = "Where"
domain = "ai.onnx"

[type_constraints.B]
types = ["tensor(bool)"]
desc = "Constrain to boolean tensors."

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to all tensor types (including bfloat)."
```
