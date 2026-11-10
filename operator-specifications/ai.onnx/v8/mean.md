```toml
name = "Mean"
domain = "ai.onnx"

[[inputs]]
name = "data_0"
type = "T"
desc = "List of tensors for mean."
option = "variadic"

[[outputs]]
name = "mean"
type = "T"
desc = "Output tensor."
```

Element-wise mean of each of the input tensors (with Numpy-style broadcasting support).
All inputs and outputs must have the same data type.
This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
