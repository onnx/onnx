```toml
name = "SplitToSequence"
domain = "ai.onnx"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input types to all tensor types."

[type_constraints.I]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain split size to integral tensor."

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
    "seq(tensor(bfloat16))",
    "seq(tensor(float16))",
    "seq(tensor(float))",
    "seq(tensor(double))",
    "seq(tensor(string))",
    "seq(tensor(bool))",
    "seq(tensor(complex64))",
    "seq(tensor(complex128))",
]
desc = "Constrain output types to all tensor types."
```
