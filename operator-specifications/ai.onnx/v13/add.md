```toml
name = "Add"
domain = "ai.onnx"

[[inputs]]
name = "A"
type = "T"
desc = "First operand."
differentiation = "differentiable"

[[inputs]]
name = "B"
type = "T"
desc = "Second operand."
differentiation = "differentiable"

[[outputs]]
name = "C"
type = "T"
desc = "Result, has same element type as two inputs"
differentiation = "differentiable"

[type_constraints.T]
types = "numeric_types_for_math_reduction_ir4"
desc = "Constrain input and output types to high-precision numeric tensors."
```
