```toml
name = "QLinearMatMul"
domain = "ai.onnx"
shape_inference = "defs::math::utils::QLinearMatMulShapeInference"

[[inputs]]
name = "a"
type = "T1"
desc = "N-dimensional quantized matrix a"
differentiation = "non_differentiable"

[[inputs]]
name = "a_scale"
type = "TS"
desc = "scale of quantized input a"
differentiation = "non_differentiable"

[[inputs]]
name = "a_zero_point"
type = "T1"
desc = "zero point of quantized input a"
differentiation = "non_differentiable"

[[inputs]]
name = "b"
type = "T2"
desc = "N-dimensional quantized matrix b"
differentiation = "non_differentiable"

[[inputs]]
name = "b_scale"
type = "TS"
desc = "scale of quantized input b"
differentiation = "non_differentiable"

[[inputs]]
name = "b_zero_point"
type = "T2"
desc = "zero point of quantized input b"
differentiation = "non_differentiable"

[[inputs]]
name = "y_scale"
type = "TS"
desc = "scale of quantized output y"
differentiation = "non_differentiable"

[[inputs]]
name = "y_zero_point"
type = "T3"
desc = "zero point of quantized output y"
differentiation = "non_differentiable"

[type_constraints.TS]
types = ["tensor(float)", "tensor(float16)", "tensor(bfloat16)"]
desc = "Constrain scales."

[type_constraints.T1]
types = [
    "tensor(int8)",
    "tensor(uint8)",
    "tensor(float8e4m3fn)",
    "tensor(float8e4m3fnuz)",
    "tensor(float8e5m2)",
    "tensor(float8e5m2fnuz)",
]
desc = "The type of input a and its zeropoint."

[type_constraints.T2]
types = [
    "tensor(int8)",
    "tensor(uint8)",
    "tensor(float8e4m3fn)",
    "tensor(float8e4m3fnuz)",
    "tensor(float8e5m2)",
    "tensor(float8e5m2fnuz)",
]
desc = "The type of input b and its zeropoint."

[type_constraints.T3]
types = [
    "tensor(int8)",
    "tensor(uint8)",
    "tensor(float8e4m3fn)",
    "tensor(float8e4m3fnuz)",
    "tensor(float8e5m2)",
    "tensor(float8e5m2fnuz)",
]
desc = "The type of the output and its zeropoint."
```
