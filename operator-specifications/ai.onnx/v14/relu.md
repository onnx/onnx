```toml
name = "Relu"
domain = "ai.onnx"

[type_constraints.T]
types = [
    "tensor(float)",
    "tensor(int32)",
    "tensor(int8)",
    "tensor(int16)",
    "tensor(int64)",
    "tensor(float16)",
    "tensor(double)",
    "tensor(bfloat16)",
]
desc = "Constrain input and output types to signed numeric tensors."

[function_bodies]
18 = """
{
   Zero = Constant <value: tensor = float {0}> ()
   ZeroCast = CastLike (Zero, X)
   Y = Max (X, ZeroCast)
}
"""
```
