```toml
name = "Shape"
domain = "ai.onnx"

[[inputs]]
name = "data"
type = "T"
desc = "An input tensor."

[[outputs]]
name = "shape"
type = "T1"
desc = "Shape of the input tensor"

[type_constraints.T]
types = "all_tensor_types"
desc = "Input tensor can be of arbitrary type."

[type_constraints.T1]
types = ["tensor(int64)"]
desc = "Constrain output to int64 tensor."
```

Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
