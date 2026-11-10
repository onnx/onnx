```toml
name = "ReduceProd"
domain = "ai.onnx"

[attributes.axes]
type = "INTS"
desc = "A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor. Accepted range is [-r, r-1] where r = rank(data)."

[attributes.keepdims]
type = "INT"
desc = "Keep the reduced dimension or not, default 1 means keep reduced dimension."
default = 1
```

Computes the product of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
