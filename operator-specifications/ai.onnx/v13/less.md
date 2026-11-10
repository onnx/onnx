```toml
name = "Less"
domain = "ai.onnx"
shape_inference = "defs::logical::utils::binaryLogicalOpInference"

[[inputs]]
name = "A"
type = "T"
desc = "First input operand for the logical operator."
differentiation = "non_differentiable"

[[inputs]]
name = "B"
type = "T"
desc = "Second input operand for the logical operator."
differentiation = "non_differentiable"

[[outputs]]
name = "C"
type = "T1"
desc = "Result tensor."
differentiation = "non_differentiable"

[type_constraints.T]
types = "all_numeric_types_ir4"
desc = "Constrain input types to all numeric tensors."

[type_constraints.T1]
types = ["tensor(bool)"]
desc = "Constrain output to boolean tensor."
```
