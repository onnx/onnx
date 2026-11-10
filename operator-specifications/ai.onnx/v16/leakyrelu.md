```toml
name = "LeakyRelu"
domain = "ai.onnx"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."

[function_bodies]
16 = """
{
   Alpha = Constant <value_float: float = @alpha> ()
   AlphaCast = CastLike (Alpha, X)
   Zero = Constant <value: tensor = float {0}> ()
   ZeroCast = CastLike (Zero, X)
   XLessThanZero = Less (X, ZeroCast)
   AlphaMulX = Mul (AlphaCast, X)
   Y = Where (XLessThanZero, AlphaMulX, X)
}
"""
```
