```toml
name = "MatMulInteger"
domain = "ai.onnx"

[[inputs]]
name = "A"
type = "T1"
desc = "N-dimensional matrix A"
differentiation = "non_differentiable"

[[inputs]]
name = "B"
type = "T2"
desc = "N-dimensional matrix B"
differentiation = "non_differentiable"

[[inputs]]
name = "a_zero_point"
type = "T1"
desc = "Zero point tensor for input 'A'. It's optional and default value is 0. It could be a scalar or N-D tensor. Scalar refers to per tensor quantization whereas N-D refers to per row quantization. If the input is 2D of shape [M, K] then zero point tensor may be an M element vector [zp_1, zp_2, ..., zp_M]. If the input is N-D tensor with shape [D1, D2, M, K] then zero point tensor may have shape [D1, D2, M, 1]. "
option = "optional"
differentiation = "non_differentiable"

[[inputs]]
name = "b_zero_point"
type = "T2"
desc = "Zero point tensor for input 'B'. It's optional and default value is 0. It could be a scalar or a N-D tensor, Scalar refers to per tensor quantization whereas N-D refers to per col quantization. If the input is 2D of shape [K, N] then zero point tensor may be an N element vector [zp_1, zp_2, ..., zp_N]. If the input is N-D tensor with shape [D1, D2, K, N] then zero point tensor may have shape [D1, D2, 1, N]. "
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "Y"
type = "T3"
desc = "Matrix multiply results from A * B"
differentiation = "non_differentiable"

[type_constraints.T1]
types = ["tensor(int8)", "tensor(uint8)"]
desc = "Constrain input A data type to 8-bit integer tensor."

[type_constraints.T2]
types = ["tensor(int8)", "tensor(uint8)"]
desc = "Constrain input B data type to 8-bit integer tensor."

[type_constraints.T3]
types = ["tensor(int32)"]
desc = "Constrain output Y data type as 32-bit integer tensor."
```

Matrix product that behaves like [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.
