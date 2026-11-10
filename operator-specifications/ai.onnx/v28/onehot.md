```toml
name = "OneHot"
domain = "ai.onnx"

[type_constraints.T1]
types = "all_numeric_types"
desc = "Constrain input to only numeric types."

[type_constraints.T2]
types = "all_numeric_types"
desc = "Constrain input to only numeric types."

[type_constraints.T3]
types = "all_tensor_types_ir4"
desc = "Constrain to any tensor type."
```
