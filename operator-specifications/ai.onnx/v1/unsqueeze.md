```toml
name = "Unsqueeze"
domain = "ai.onnx"

[attributes.axes]
type = "INTS"
desc = "List of non-negative integers, indicate the dimensions to be inserted"
required = true

[[inputs]]
name = "data"
type = "T"
desc = "Original tensor"

[[outputs]]
name = "expanded"
type = "T"
desc = "Reshaped tensor with same data as input."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to all tensor types."
```

Insert single-dimensional entries to the shape of a tensor.
Takes one required argument `axes`, a list of dimensions that will be inserted.
Dimension indices in `axes` are as seen in the output tensor. For example:
  Given a tensor such that tensor with shape [3, 4, 5], then
  Unsqueeze(tensor, axes=[0, 4]) has shape [1, 3, 4, 5, 1]
