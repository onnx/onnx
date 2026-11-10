```toml
name = "LpNormalization"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "The axis on which to apply normalization, -1 mean last axis."
default = -1

[attributes.p]
type = "INT"
desc = "The order of the normalization, only 1 or 2 are supported."
default = 2

[[inputs]]
name = "input"
type = "T"
desc = "Input matrix"
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Matrix after normalization"
differentiation = "differentiable"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Given a matrix, apply Lp-normalization along the provided axis.
The output is computed as: `output = input / Lp_norm(input, axis)`.
When the Lp norm is zero (i.e., all elements along the axis are zero),
the output is defined to be zero to avoid division by zero.
