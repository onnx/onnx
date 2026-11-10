```toml
name = "Upsample"
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
desc = "The scale array along each dimension. It takes value greater than or equal to 1. The number of elements of 'scales' should be the same as the rank of input 'X'."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input 'X' and output 'Y' to all tensor types."
```
