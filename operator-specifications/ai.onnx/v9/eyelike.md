```toml
name = "EyeLike"
domain = "ai.onnx"

[attributes.dtype]
type = "INT"
desc = "(Optional) The data type for the elements of the output tensor. If not specified,the data type of the input tensor T1 is used. If input tensor T1 is also notspecified, then type defaults to 'float'."

[attributes.k]
type = "INT"
desc = "(Optional) Index of the diagonal to be populated with ones. Default is 0. If T2 is the output, this op sets T2[i, i+k] = 1. k = 0 populates the main diagonal, k > 0 populates an upper diagonal,  and k < 0 populates a lower diagonal."
default = 0

[[inputs]]
name = "input"
type = "T1"
desc = "2D input tensor to copy shape, and optionally, type information from."

[[outputs]]
name = "output"
type = "T2"
desc = "Output tensor, same shape as input tensor T1."

[type_constraints.T1]
types = [
    "tensor(float16)",
    "tensor(float)",
    "tensor(double)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
    "tensor(uint8)",
    "tensor(uint16)",
    "tensor(uint32)",
    "tensor(uint64)",
    "tensor(bool)",
]
desc = "Constrain input types. Strings and complex are not supported."

[type_constraints.T2]
types = [
    "tensor(float16)",
    "tensor(float)",
    "tensor(double)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
    "tensor(uint8)",
    "tensor(uint16)",
    "tensor(uint32)",
    "tensor(uint64)",
    "tensor(bool)",
]
desc = "Constrain output types. Strings and complex are not supported."
```

Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
same as the input tensor. The data type can be specified by the 'dtype' argument. If
'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
