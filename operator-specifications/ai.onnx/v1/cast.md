```toml
name = "Cast"
domain = "ai.onnx"
shape_inference = false

[attributes.to]
type = "STRING"
desc = "The data type to which the elements of the input tensor are cast. Strictly must be one of the types from DataType enum in TensorProto"
required = true

[[inputs]]
name = "input"
type = "T1"
desc = "Input tensor to be cast."

[[outputs]]
name = "output"
type = "T2"
desc = "Output tensor with the same shape as input with type specified by the 'to' argument"

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
desc = "Constrain input types. Casting from strings and complex are not supported."

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
desc = "Constrain output types. Casting to strings and complex are not supported."
```

The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.
NOTE: Casting to and from strings is not supported yet.
