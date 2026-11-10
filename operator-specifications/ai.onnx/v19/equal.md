```toml
name = "Equal"
domain = "ai.onnx"

[type_constraints.T]
types = [
    "tensor(bool)",
    "tensor(uint8)",
    "tensor(uint16)",
    "tensor(uint32)",
    "tensor(uint64)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int32)",
    "tensor(int64)",
    "tensor(float16)",
    "tensor(float)",
    "tensor(double)",
    "tensor(bfloat16)",
    "tensor(string)",
]
desc = "Constrain input types to all (non-complex) tensors."

[type_constraints.T1]
types = ["tensor(bool)"]
desc = "Constrain output to boolean tensor."
```
