```toml
name = "Resize"
domain = "ai.onnx"

[attributes.mode]
type = "STRING"
desc = "Two interpolation modes: nearest (default), and linear (including bilinear, trilinear, etc)"
default = "nearest"

[[inputs]]
name = "X"
type = "T"
desc = "N-D tensor"

[[inputs]]
name = "scales"
type = "tensor(float)"
desc = "The scale array along each dimension. It takes value greater than 0. If it's less than 1, it's sampling down, otherwise, it's upsampling. The number of elements of 'scales' should be the same as the rank of input 'X'."

[[outputs]]
name = "Y"
type = "T"
desc = "N-D tensor after resizing"

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input 'X' and output 'Y' to all tensor types."
```

Resize the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
