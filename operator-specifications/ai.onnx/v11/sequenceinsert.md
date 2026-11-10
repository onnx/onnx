```toml
name = "SequenceInsert"
domain = "ai.onnx"

[[inputs]]
name = "input_sequence"
type = "S"
desc = "Input sequence."

[[inputs]]
name = "tensor"
type = "T"
desc = "Input tensor to be inserted into the input sequence."

[[inputs]]
name = "position"
type = "I"
desc = "Position in the sequence where the new tensor is inserted. It is optional and default is to insert to the back of the sequence. Negative value means counting positions from the back. Accepted range in `[-n, n]`, where `n` is the number of tensors in 'input_sequence'. It is an error if any of the index values are out of bounds. It must be a scalar(tensor of empty shape)."
option = "optional"

[[outputs]]
name = "output_sequence"
type = "S"
desc = "Output sequence that contains the inserted tensor at given position."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain to any tensor type."

[type_constraints.S]
types = [
    "seq(tensor(uint8))",
    "seq(tensor(uint16))",
    "seq(tensor(uint32))",
    "seq(tensor(uint64))",
    "seq(tensor(int8))",
    "seq(tensor(int16))",
    "seq(tensor(int32))",
    "seq(tensor(int64))",
    "seq(tensor(float16))",
    "seq(tensor(float))",
    "seq(tensor(double))",
    "seq(tensor(string))",
    "seq(tensor(bool))",
    "seq(tensor(complex64))",
    "seq(tensor(complex128))",
]
desc = "Constrain to any tensor type."

[type_constraints.I]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain position to integral tensor. It must be a scalar(tensor of empty shape)."
```

Outputs a tensor sequence that inserts 'tensor' into 'input_sequence' at 'position'.
'tensor' must have the same data type as 'input_sequence'.
Accepted range for 'position' is in `[-n, n]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it inserts 'tensor' to the back of 'input_sequence'.
