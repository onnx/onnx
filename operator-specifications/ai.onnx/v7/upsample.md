```toml
name = "Upsample"
domain = "ai.onnx"
support_level = "common"

[attributes.mode]
type = "STRING"
desc = "Two interpolation modes: nearest (default), and linear (including bilinear, trilinear, etc)"
default = "nearest"

[attributes.scales]
type = "FLOATS"
desc = "The scale array along each dimension. It takes value greater than or equal to 1. The number of elements of 'scales' should be the same as the rank of input 'X'."
required = true

[[inputs]]
name = "X"
type = "T"
desc = "N-D tensor"

[[outputs]]
name = "Y"
type = "T"
desc = "N-D tensor after resizing"

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to all tensor types."
```

Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
