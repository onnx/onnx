```toml
name = "Pow"
domain = "ai.onnx"

[type_constraints.T]
types = [
    "tensor(int32)",
    "tensor(int64)",
    "tensor(float16)",
    "tensor(float)",
    "tensor(double)",
    "tensor(bfloat16)",
]
desc = "Constrain input X and output types to float/int tensors."

[type_constraints.T1]
types = "all_numeric_types_ir4"
desc = "Constrain input Y types to float/int tensors."
```
