```toml
name = "Gemm"
domain = "ai.onnx"

[type_constraints.T]
types = "numeric_types_for_math_reduction"
desc = "Constrain input and output types to float/int tensors."
```
