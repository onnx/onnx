```toml
name = "Neg"
domain = "ai.onnx"
attributes = {}

[type_constraints.T]
types = [
    "tensor(float)",
    "tensor(int32)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int64)",
    "tensor(float16)",
    "tensor(double)",
]
desc = "Constrain input and output types to signed numeric tensors."
```
