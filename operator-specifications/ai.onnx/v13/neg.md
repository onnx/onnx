```toml
name = "Neg"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[[inputs]]
name = "X"
type = "T"
desc = "Input tensor"
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor"
differentiation = "differentiable"

[type_constraints.T]
types = [
    "tensor(float)",
    "tensor(int32)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int64)",
    "tensor(float16)",
    "tensor(double)",
    "tensor(bfloat16)",
]
desc = "Constrain input and output types to signed numeric tensors."
```
