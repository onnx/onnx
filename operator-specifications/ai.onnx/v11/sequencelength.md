```toml
name = "SequenceLength"
domain = "ai.onnx"

[[inputs]]
name = "input_sequence"
type = "S"
desc = "Input sequence."

[[outputs]]
name = "length"
type = "I"
desc = "Length of input sequence. It must be a scalar(tensor of empty shape)."

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
types = ["tensor(int64)"]
desc = "Constrain output to integral tensor. It must be a scalar(tensor of empty shape)."
```

Produces a scalar(tensor of empty shape) containing the number of tensors in 'input_sequence'.
