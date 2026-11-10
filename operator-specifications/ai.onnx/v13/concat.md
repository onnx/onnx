```toml
name = "Concat"
domain = "ai.onnx"

[[inputs]]
name = "inputs"
type = "T"
desc = "List of tensors for concatenation"
option = "variadic"
differentiation = "differentiable"

[[outputs]]
name = "concat_result"
type = "T"
desc = "Concatenated tensor"
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain output types to any tensor type."
```
