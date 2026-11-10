```toml
name = "Gemm"
domain = "ai.onnx"
shape_inference = "defs::math::utils::GemmShapeInference"
function_body_builders = { 29 = "defs::math::utils::BuildContextDependentFunctionBodyGemm" }

[attributes.alpha]
type = "FLOAT"
desc = "Scalar multiplier for the product of input tensors A * B."
default = 1.0

[attributes.beta]
type = "FLOAT"
desc = "Scalar multiplier for input tensor C."
default = 1.0

[attributes.transA]
type = "INT"
desc = "Whether to swap the last two axes of A. Has no effect on rank-1 A."
default = 0

[attributes.transB]
type = "INT"
desc = "Whether B should be transposed"
default = 0

[[inputs]]
name = "A"
type = "T"
desc = "Input tensor A. Rank >= 1 tensor. After applying transA, its shape is (..., K)."
differentiation = "differentiable"

[[inputs]]
name = "B"
type = "T"
desc = "Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero."
differentiation = "differentiable"

[[inputs]]
name = "C"
type = "T"
desc = "Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C must be unidirectionally broadcastable to the output shape (..., N)."
option = "optional"
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor of shape (..., N)."
differentiation = "differentiable"
```

Generalized matrix multiplication: `Y = alpha * A' * B' + beta * C`.
A has rank at least 1 and B has rank 2. A' is A with its last two axes swapped
when transA is non-zero; for rank-1 A, transA has no effect.
B' is B transposed when transB is non-zero.
If A' has shape (..., K) and B' has shape (K, N), Y has shape (..., N):
each vector along A's contraction axis is multiplied by the same matrix B'.
The leading dimensions of A' are preserved, including when there are none.
C must be unidirectionally broadcastable to Y. If C is omitted, it is treated
as scalar zero. When beta is zero, C does not contribute to the result.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A' * B'); for more details please check [the doc](Broadcasting.md).
This operator has **optional** inputs/outputs. See [the doc](IR.md) for more details about the representation of optional arguments. An empty string may be used in the place of an actual argument's name to indicate a missing argument. Trailing optional arguments (those not followed by an argument that is present) may also be simply omitted.
