```toml
name = "Reshape"
domain = "ai.onnx"

[attributes.allowzero]
type = "INT"
desc = "(Optional) By default, when any value in the 'shape' input is equal to zero the corresponding dimension value is copied from the input tensor dynamically. allowzero=1 indicates that if any value in the 'shape' input is set to zero, the zero value is honored, similar to NumPy."
default = 0
```

Reshape the input tensor similar to numpy.reshape.
First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
At most one dimension of the new shape can be -1. In this case, the value is
inferred from the size of the tensor and the remaining dimensions. A dimension
could also be 0, in which case the actual dimension value is unchanged (i.e. taken
from the input tensor). If 'allowzero' is set, and the new shape includes 0, the
dimension will be set explicitly to zero (i.e. not taken from input tensor).
Shape (second input) could be an empty shape, which means converting to a scalar.
The input tensor's shape and the output tensor's shape are required to have the same number of elements.

If the attribute 'allowzero' is set, it is invalid for the specified shape to
contain both a zero value and -1, as the value of the dimension corresponding
to -1 cannot be determined uniquely.
