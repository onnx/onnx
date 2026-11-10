```toml
name = "ReduceMin"
domain = "ai.onnx"

[attributes.axes]
type = "INTS"
desc = "A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor. Accepted range is [-r, r-1] where r = rank(data)."

[attributes.keepdims]
type = "INT"
desc = "Keep the reduced dimension or not, default 1 means keep reduced dimension."
default = 1
```
