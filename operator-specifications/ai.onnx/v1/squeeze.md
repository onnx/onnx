```toml
name = "Squeeze"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::SqueezeShapeInference_opset1"

[attributes.axes]
type = "INTS"
desc = "List of non-negative integers, indicate the dimensions to squeeze."

[[inputs]]
name = "data"
type = "T"
desc = "Tensors with at least max(dims) dimensions."

[[outputs]]
name = "squeezed"
type = "T"
desc = "Reshaped tensor with same data as input."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to all tensor types."
```

Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
