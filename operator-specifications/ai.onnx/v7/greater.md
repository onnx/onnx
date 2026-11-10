```toml
name = "Greater"
domain = "ai.onnx"
attributes = {}

[[inputs]]
name = "A"
type = "T"
desc = "First input operand for the logical operator."

[[inputs]]
name = "B"
type = "T"
desc = "Second input operand for the logical operator."
```

Returns the tensor resulted from performing the `greater` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
