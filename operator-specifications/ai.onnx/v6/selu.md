```toml
name = "Selu"
domain = "ai.onnx"

[attributes.alpha]
type = "FLOAT"
desc = "Coefficient of SELU default to 1.67326319217681884765625 (i.e., float32 approximation of 1.6732632423543772848170429916717)."
default = 1.6732632

[attributes.gamma]
type = "FLOAT"
desc = "Coefficient of SELU default to 1.05070102214813232421875 (i.e., float32 approximation of 1.0507009873554804934193349852946)."
default = 1.050701

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
   Gamma = Constant <value_float: float = @gamma> ()
   GammaCast = CastLike (Gamma, X)
   Zero = Constant <value: tensor = float {0}> ()
   ZeroCast = CastLike (Zero, X)
   ExpX = Exp (X)
   AlphaMulExpX = Mul (AlphaCast, ExpX)
   AlphaMulExpXSubAlpha = Sub (AlphaMulExpX, AlphaCast)
   Neg = Mul (GammaCast, AlphaMulExpXSubAlpha)
   Pos = Mul (GammaCast, X)
   XLessThanZero = Less (X, ZeroCast)
   Y = Where (XLessThanZero, Neg, Pos)
}
"""
```
