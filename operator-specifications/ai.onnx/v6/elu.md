```toml
name = "Elu"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[attributes.alpha]
type = "FLOAT"
desc = "Coefficient of ELU."
default = 1.0

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

[function_bodies]
18 = """
{
   Alpha = Constant <value_float: float = @alpha> ()
   AlphaCast = CastLike (Alpha, X)
   Zero = Constant <value: tensor = float {0}> ()
   ZeroCast = CastLike (Zero, X)
   One = Constant <value: tensor = float {1}> ()
   OneCast = CastLike (One, X)
   XLessThanZero = Less (X, ZeroCast)
   ExpX = Exp (X)
   ExpXSubOne = Sub (ExpX, OneCast)
   AlphaMulExpXSubOne = Mul (AlphaCast, ExpXSubOne)
   Y = Where (XLessThanZero, AlphaMulExpXSubOne, X)
}
"""
```
