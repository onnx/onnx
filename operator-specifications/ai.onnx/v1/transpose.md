```toml
name = "Transpose"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::TransposeShapeInference_opset1"

[attributes.perm]
type = "INTS"
desc = "A list of integers. By default, reverse the dimensions, otherwise permute the axes according to the values given."

[[inputs]]
name = "data"
type = "T"
desc = "An input tensor."

[[outputs]]
name = "transposed"
type = "T"
desc = "Transposed output."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to all tensor types."
```

Returns a transpose of the input tensor. (Similar to `numpy.transpose`).
The optional attribute `perm` specifies the permutation of the axes of the
input tensor. `perm` must contain each axis index in `[0, n-1]` exactly once,
so its length is equal to the rank `n` of the input tensor.

Axis `i` of the output tensor corresponds to axis `perm[i]` of the input tensor.

If the attribute is omitted, its default value is `(n-1, ..., 0)`, where `n`
is the rank of the input tensor (that is, the dimensions are reversed).

For example, when perm=(1, 0, 2), given an input tensor of shape (1, 2, 3),
the output shape will be (2, 1, 3).
When perm=(1, 2, 0), given an input tensor of shape (1, 2, 3),
the output shape will be (2, 3, 1).
A 0-D or 1-D input is valid; in those cases the output has the same shape
as the input.
