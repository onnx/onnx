```toml
name = "ArgMax"
domain = "ai.onnx"
shape_inference = "defs::reduction::utils::ArgReduceShapeInference_opset11"

[attributes.axis]
type = "INT"
desc = "The axis in which to compute the arg indices. Accepted range is [-r, r-1] where r = rank(data)."
default = 0

[attributes.keepdims]
type = "INT"
desc = "Keep the reduced dimension or not, default 1 means keep reduced dimension."
default = 1
```

Computes the indices of the max elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulting tensor has the reduced dimension pruned.
The input tensor must not be empty.
The type of the output tensor is integer.
