```toml
name = "RandomUniformLike"
domain = "ai.onnx"

[type_constraints.T1]
types = "all_tensor_types_ir4"
desc = "Constrain to any tensor type. If the dtype attribute is not provided this must be a valid output type."

[type_constraints.T2]
types = "all_float_types_ir4"
desc = "Constrain output types to float tensors."
```
