```toml
name = "Min"
domain = "ai.onnx"

[[inputs]]
name = "data_0"
type = "T"
desc = "List of tensors for min."
option = "variadic"

[[outputs]]
name = "min"
type = "T"
desc = "Output tensor."
```

Element-wise min of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
