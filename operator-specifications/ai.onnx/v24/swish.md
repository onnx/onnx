```toml
name = "Swish"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[attributes.alpha]
type = "FLOAT"
desc = "Coefficient to multiply with input before sigmoid."
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

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."

[function_bodies]
24 = """
{
   Alpha = Constant <value_float: float = @alpha> ()
   AlphaCast = CastLike (Alpha, X)
   AlphaMulX = Mul (AlphaCast, X)
   SigmoidAlphaMulX = Sigmoid (AlphaMulX)
   Y = Mul (X, SigmoidAlphaMulX)
}
"""
```

Swish function takes one input data (Tensor<T>) and produces one output data (Tensor<T>) of the same shape,
where $Swish(x) = x * sigmoid(alpha * x)$.
