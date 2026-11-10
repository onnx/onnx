```toml
name = "ReduceMean"
domain = "ai.onnx"
shape_inference = "defs::reduction::utils::ReduceShapeInference_opset1"

[attributes.axes]
type = "INTS"
desc = "A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor."

[attributes.keepdims]
type = "INT"
desc = "Keep the reduced dimension or not, default 1 means keep reduced dimension."
default = 1

[[inputs]]
name = "data"
type = "T"
desc = "An input tensor."

[[outputs]]
name = "reduced"
type = "T"
desc = "Reduced output tensor."

[type_constraints.T]
types = "numeric_types_for_math_reduction"
desc = "Constrain input and output types to high-precision numeric tensors."
```

Computes the mean of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned. Input tensors of rank zero are
valid. Reduction over an empty set of values yields undefined.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.
