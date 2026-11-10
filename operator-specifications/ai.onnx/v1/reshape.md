```toml
name = "Reshape"
domain = "ai.onnx"
shape_inference = false

[attributes.consumed_inputs]
type = "INTS"
desc = "legacy optimization attribute."

[attributes.shape]
type = "INTS"
desc = "New shape"

[[inputs]]
name = "data"
type = "T"
desc = "An input tensor."

[[outputs]]
name = "reshaped"
type = "T"
desc = "Reshaped data."

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Reshape the input tensor similar to numpy.reshape.
It takes a tensor as input and an argument `shape`. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor). Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.
