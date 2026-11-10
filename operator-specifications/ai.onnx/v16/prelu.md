```toml
name = "PRelu"
domain = "ai.onnx"

[type_constraints.T]
types = "numeric_types_for_math_reduction_ir4"
desc = "Constrain input and output types to float/int tensors."

[function_bodies]
16 = """
{
   Zero = Constant <value: tensor = float {0}> ()
   ZeroCast = CastLike (Zero, X)
   XLessThanZero = Less (X, ZeroCast)
   SlopeMulX = Mul (slope, X)
   Y = Where (XLessThanZero, SlopeMulX, X)
}
"""
```

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).
