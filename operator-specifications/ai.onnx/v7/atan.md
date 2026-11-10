```toml
name = "Atan"
domain = "ai.onnx"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor"
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "The arctangent of the input tensor computed element-wise"
differentiation = "differentiable"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.
