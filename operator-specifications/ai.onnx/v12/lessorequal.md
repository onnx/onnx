```toml
name = "LessOrEqual"
domain = "ai.onnx"
shape_inference = false

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
types = "all_numeric_types"
desc = "Constrain input types to all numeric tensors."

[type_constraints.T1]
types = ["tensor(bool)"]
desc = "Constrain output to boolean tensor."

[function_bodies]
12 = """
{
   O1 = Less (A, B)
   O2 = Equal (A, B)
   C = Or (O1, O2)
}
"""
```

Returns the tensor resulted from performing the `less_equal` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
