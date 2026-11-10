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
type = "tensor(float)"
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
type = "tensor(float)"
desc = "scale of quantized input b"
differentiation = "non_differentiable"

[[inputs]]
name = "b_zero_point"
type = "T2"
desc = "zero point of quantized input b"
differentiation = "non_differentiable"

[[inputs]]
name = "y_scale"
type = "tensor(float)"
desc = "scale of quantized output y"
differentiation = "non_differentiable"

[[inputs]]
name = "y_zero_point"
type = "T3"
desc = "zero point of quantized output y"
differentiation = "non_differentiable"

[[outputs]]
name = "y"
type = "T3"
desc = "Quantized matrix multiply results from a * b"
differentiation = "non_differentiable"

[type_constraints.T1]
types = ["tensor(int8)", "tensor(uint8)"]
desc = "Constrain input a and its zero point data type to 8-bit integer tensor."

[type_constraints.T2]
types = ["tensor(int8)", "tensor(uint8)"]
desc = "Constrain input b and its zero point data type to 8-bit integer tensor."

[type_constraints.T3]
types = ["tensor(int8)", "tensor(uint8)"]
desc = "Constrain output y and its zero point data type to 8-bit integer tensor."
```

Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output,
and computes the quantized output. The quantization formula is y = saturate((x / y_scale) + y_zero_point).
For (x / y_scale), it is rounding to nearest ties to even. Refer to https://en.wikipedia.org/wiki/Rounding for details.
Scale and zero point must have same shape. They must be either scalar (per tensor) or N-D tensor
(per row for 'a' and per column for 'b'). Scalar refers to per tensor quantization whereas N-D refers to per row
or per column quantization. If the input is 2D of shape [M, K] then zero point and scale tensor may be
an M element vector [v_1, v_2, ..., v_M] for per row quantization and K element vector of shape [v_1, v_2, ..., v_K]
for per column quantization. If the input is N-D tensor with shape [D1, D2, M, K] then zero point and scale tensor may
have shape [D1, D2, M, 1] for per row quantization and shape [D1, D2, 1, K] for per column quantization.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.
