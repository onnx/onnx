```toml
name = "PRelu"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[[inputs]]
name = "X"
type = "T"
desc = "Input tensor"
differentiation = "differentiable"

[[inputs]]
name = "slope"
type = "T"
desc = "Slope tensor. The shape of slope can be smaller than first input X; if so, its shape must be unidirectional broadcastable to X"
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor (same size as X)"
differentiation = "differentiable"

[type_constraints.T]
types = "numeric_types_for_math_reduction"
desc = "Constrain input and output types to float/int tensors."
```
