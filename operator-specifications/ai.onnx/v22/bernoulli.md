```toml
name = "Bernoulli"
domain = "ai.onnx"

[type_constraints.T1]
types = "all_float_types_ir4"
desc = "Constrain input types to float tensors."

[type_constraints.T2]
types = "all_non_complex_numeric_types_plus_bool_ir4"
desc = "Constrain output types to all numeric tensors and bool tensors."
```
