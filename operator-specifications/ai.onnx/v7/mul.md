```toml
name = "Mul"
domain = "ai.onnx"
attributes = {}

[[inputs]]
name = "A"
type = "T"
desc = "First operand."

[[inputs]]
name = "B"
type = "T"
desc = "Second operand."

[[outputs]]
name = "C"
type = "T"
desc = "Result, has same element type as two inputs"
```

Performs element-wise binary multiplication (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
