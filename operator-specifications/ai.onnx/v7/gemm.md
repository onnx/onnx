```toml
name = "Gemm"
domain = "ai.onnx"

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
desc = "Whether A should be transposed"
default = 0

[attributes.transB]
type = "INT"
desc = "Whether B should be transposed"
default = 0

[[inputs]]
name = "A"
type = "T"
desc = "Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero."

[[inputs]]
name = "B"
type = "T"
desc = "Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero."

[[inputs]]
name = "C"
type = "T"
desc = "Input tensor C. The shape of C should be unidirectional broadcastable to (M, N)."

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor of shape (M, N)."
```

General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3

A' = transpose(A) if transA else A

B' = transpose(B) if transB else B

Compute Y = alpha * A' * B' + beta * C, where input tensor A has shape (M, K) or (K, M),
input tensor B has shape (K, N) or (N, K), input tensor C is broadcastable to shape (M, N),
and output tensor Y has shape (M, N). A will be transposed before doing the
computation if attribute transA is non-zero, same for B and transB.
This operator supports **unidirectional broadcasting** (tensor C should be unidirectional broadcastable to tensor A * B); for more details please check [the doc](Broadcasting.md).
