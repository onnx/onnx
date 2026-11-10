```toml
name = "ArgMax"
domain = "ai.onnx"
shape_inference = "defs::reduction::utils::ArgReduceShapeInference_opset1"

[attributes.axis]
type = "INT"
desc = "The axis in which to compute the arg indices."
default = 0

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
type = "tensor(int64)"
desc = "Reduced output tensor with integer data type."

[type_constraints.T]
types = "all_numeric_types"
desc = "Constrain input and output types to all numeric tensors."
```

Computes the indices of the max elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
The type of the output tensor is integer.
