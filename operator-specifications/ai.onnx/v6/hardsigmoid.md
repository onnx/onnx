```toml
name = "HardSigmoid"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[attributes.alpha]
type = "FLOAT"
desc = "Value of alpha."
default = 0.2

[attributes.beta]
type = "FLOAT"
desc = "Value of beta."
default = 0.5

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
   Beta = Constant <value_float: float = @beta> ()
   BetaCast = CastLike (Beta, X)
   Zero = Constant <value: tensor = float {0}> ()
   ZeroCast = CastLike (Zero, X)
   One = Constant <value: tensor = float {1}> ()
   OneCast = CastLike (One, X)
   AlphaMulX = Mul (X, AlphaCast)
   AlphaMulXAddBeta = Add (AlphaMulX, BetaCast)
   MinOneOrAlphaMulXAddBeta = Min (AlphaMulXAddBeta, OneCast)
   Y = Max (MinOneOrAlphaMulXAddBeta, ZeroCast)
}
"""
```
