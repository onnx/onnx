```toml
name = "Pow"
domain = "ai.onnx"
attributes = {}

[[inputs]]
name = "X"
type = "T"
desc = "First operand, base of the exponent."

[[inputs]]
name = "Y"
type = "T"
desc = "Second operand, power of the exponent."

[[outputs]]
name = "Z"
type = "T"
desc = "Output tensor."
```

Pow takes input data (Tensor<T>) and exponent Tensor, and
produces one output data (Tensor<T>) where the function `f(x) = x^exponent`,
is applied to the data tensor elementwise.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
