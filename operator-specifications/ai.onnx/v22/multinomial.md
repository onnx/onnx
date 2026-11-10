```toml
name = "Multinomial"
domain = "ai.onnx"

[type_constraints.T1]
types = "all_float_types_ir4"
desc = "Constrain input types to float tensors."

[type_constraints.T2]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain output types to integral tensors."
```
