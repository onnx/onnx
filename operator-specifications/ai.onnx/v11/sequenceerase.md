```toml
name = "SequenceErase"
domain = "ai.onnx"

[[inputs]]
name = "input_sequence"
type = "S"
desc = "Input sequence."

[[inputs]]
name = "position"
type = "I"
desc = "Position of the tensor in the sequence. Negative value means counting positions from the back. Accepted range in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'. It is an error if any of the index values are out of bounds. It must be a scalar(tensor of empty shape)."
option = "optional"

[[outputs]]
name = "output_sequence"
type = "S"
desc = "Output sequence that has the tensor at the specified position removed."

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

Outputs a tensor sequence that removes the tensor at 'position' from 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
'position' is optional, by default it erases the last tensor from 'input_sequence'.
