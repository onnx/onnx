```toml
name = "BitwiseNot"
domain = "ai.onnx"

[[inputs]]
name = "X"
type = "T"
desc = "Input tensor"
differentiation = "non_differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor"
differentiation = "non_differentiable"

[type_constraints.T]
types = [
    "tensor(uint8)",
    "tensor(uint16)",
    "tensor(uint32)",
    "tensor(uint64)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
]
desc = "Constrain input/output to integer tensors."
```

Returns the bitwise not of the input tensor element-wise.
