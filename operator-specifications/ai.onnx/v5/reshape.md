```toml
name = "Reshape"
domain = "ai.onnx"
attributes = {}

[[inputs]]
name = "data"
type = "T"
desc = "An input tensor."

[[inputs]]
name = "shape"
type = "tensor(int64)"
desc = "Specified shape for output."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to all tensor types."
```

Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor). Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.
