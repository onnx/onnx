```toml
name = "MatMul"
domain = "ai.onnx"
shape_inference = "defs::math::utils::MatMulShapeInference_opset9"

[type_constraints.T]
types = "numeric_types_for_math_reduction"
desc = "Constrain input and output types to float/int tensors."
```
