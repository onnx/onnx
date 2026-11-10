```toml
name = "Gemm"
domain = "ai.onnx"

[attributes.alpha]
type = "FLOAT"
desc = "Scalar multiplier for the product of input tensors A * B, the default value is 1.0."
default = 1.0

[attributes.beta]
type = "FLOAT"
desc = "Scalar multiplier for input tensor C, the default value is 1.0."
default = 1.0

[attributes.broadcast]
type = "INT"
desc = "Whether C should be broadcasted"
default = 0

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
desc = "Input tensor A"

[[inputs]]
name = "B"
type = "T"
desc = "Input tensor B"

[[inputs]]
name = "C"
type = "T"
desc = "Input tensor C, can be inplace."

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor."

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

General Matrix multiplication:
https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
Compute Y = alpha * A * B + beta * C, where input tensor A has
dimension (M X K), input tensor B has dimension (K X N), input tensor C and
output tensor Y have dimension (M X N).
If attribute broadcast is non-zero, input tensor C will be broadcasted to match
the dimension requirement. A will be transposed before doing the computation
if attribute transA is non-zero, same for B and transB.
