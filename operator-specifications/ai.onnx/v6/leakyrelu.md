```toml
name = "LeakyRelu"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[attributes.alpha]
type = "FLOAT"
desc = "Coefficient of leakage."
default = 0.01

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
```
