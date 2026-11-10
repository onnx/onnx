```toml
name = "IsInf"
domain = "ai.onnx"

[type_constraints.T1]
types = "all_float_types_ir9"
desc = "Constrain input types to float tensors."

[type_constraints.T2]
types = ["tensor(bool)"]
desc = "Constrain output types to boolean tensors."
```
