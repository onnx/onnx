```toml
name = "Det"
domain = "ai.onnx"

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

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to floating-point tensors."
```

Det calculates determinant of a square matrix or batches of square matrices.
Det takes one input tensor of shape `[*, M, M]`, where `*` is zero or more batch dimensions,
and the inner-most 2 dimensions form square matrices.
The output is a tensor of shape `[*]`, containing the determinants of all input submatrices.
e.g., When the input is 2-D, the output is a scalar(shape is empty: `[]`).
