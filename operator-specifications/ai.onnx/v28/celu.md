```toml
name = "Celu"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."

[function_bodies]
28 = """
{
   Alpha = Constant <value_float: float = @alpha> ()
   AlphaCast = CastLike (Alpha, X)
   XScaled = Div (X, AlphaCast)
   EluResult = Elu <alpha: float = 1> (XScaled)
   Y = Mul (AlphaCast, EluResult)
}
"""
```
