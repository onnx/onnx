```toml
name = "Mish"
domain = "ai.onnx"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input X and output types to float tensors."

[function_bodies]
22 = """
{
   Softplus_X = Softplus (X)
   TanHSoftplusX = Tanh (Softplus_X)
   Y = Mul (X, TanHSoftplusX)
}
"""
```
