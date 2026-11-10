```toml
name = "Mish"
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
desc = "Constrain input X and output types to float tensors."

[function_bodies]
18 = """
{
   Softplus_X = Softplus (X)
   TanHSoftplusX = Tanh (Softplus_X)
   Y = Mul (X, TanHSoftplusX)
}
"""
```

Mish: A Self Regularized Non-Monotonic Neural Activation Function.

Perform the linear unit element-wise on the input tensor X using formula:

```
mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
```
