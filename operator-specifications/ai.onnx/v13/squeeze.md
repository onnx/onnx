```toml
name = "Squeeze"
domain = "ai.onnx"
attributes = {}

[[inputs]]
name = "data"
type = "T"
desc = "Tensors with at least max(dims) dimensions."
differentiation = "differentiable"

[[inputs]]
name = "axes"
type = "tensor(int64)"
desc = "List of integers indicating the dimensions to squeeze. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data)."
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "squeezed"
type = "T"
desc = "Reshaped tensor with same data as input."
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to all tensor types."
```

Remove single-dimensional entries from the shape of a tensor.
Takes an input `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
