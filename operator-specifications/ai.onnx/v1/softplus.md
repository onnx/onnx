```toml
name = "Softplus"
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
desc = "Constrain input and output types to float tensors."

[function_bodies]
18 = """
{
   exp_x = Exp (X)
   one = Constant <value: tensor = float {1}> ()
   one_cast = CastLike (one, X)
   exp_x_add_one = Add (exp_x, one_cast)
   Y = Log (exp_x_add_one)
}
"""
```

Softplus takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the softplus function, y = ln(exp(x) + 1), is applied to
the tensor elementwise.
