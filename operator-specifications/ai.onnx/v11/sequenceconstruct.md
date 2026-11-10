```toml
name = "SequenceConstruct"
domain = "ai.onnx"

[[inputs]]
name = "inputs"
type = "T"
desc = "Tensors."
option = "variadic"

[[outputs]]
name = "output_sequence"
type = "S"
desc = "Sequence enclosing the input tensors."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input types to any tensor type."

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
desc = "Constrain output types to any tensor type."
```

Construct a tensor sequence containing 'inputs' tensors.
All tensors in 'inputs' must have the same data type.
