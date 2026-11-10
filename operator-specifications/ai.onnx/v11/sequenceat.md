```toml
name = "SequenceAt"
domain = "ai.onnx"
shape_inference = "defs::sequence::utils::SequenceAtShapeInference_opset11"

[[inputs]]
name = "input_sequence"
type = "S"
desc = "Input sequence."

[[inputs]]
name = "position"
type = "I"
desc = "Position of the tensor in the sequence. Negative value means counting positions from the back. Accepted range in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'. It is an error if any of the index values are out of bounds. It must be a scalar(tensor of empty shape)."

[[outputs]]
name = "tensor"
type = "T"
desc = "Output tensor at the specified position in the input sequence."

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

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain to any tensor type."

[type_constraints.I]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain position to integral tensor. It must be a scalar(tensor of empty shape)."
```

Outputs a tensor copy from the tensor at 'position' in 'input_sequence'.
Accepted range for 'position' is in `[-n, n - 1]`, where `n` is the number of tensors in 'input_sequence'.
Negative value means counting positions from the back.
