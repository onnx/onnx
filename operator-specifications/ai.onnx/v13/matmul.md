```toml
name = "MatMul"
domain = "ai.onnx"
shape_inference = "defs::math::utils::MatMulShapeInference_opset13"

[[inputs]]
name = "A"
type = "T"
desc = "N-dimensional matrix A"
differentiation = "differentiable"

[[inputs]]
name = "B"
type = "T"
desc = "N-dimensional matrix B"
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "Matrix multiply results from A * B"
differentiation = "differentiable"

[type_constraints.T]
types = "numeric_types_for_math_reduction_ir4"
desc = "Constrain input and output types to float/int tensors."
```
