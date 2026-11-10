```toml
name = "ThresholdedRelu"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[attributes.alpha]
type = "FLOAT"
desc = "Threshold value"
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
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."

[function_bodies]
18 = """
{
   Alpha = Constant <value_float: float = @alpha> ()
   AlphaCast = CastLike (Alpha, X)
   Zero = Constant <value: tensor = float {0}> ()
   ZeroCast = CastLike (Zero, X)
   AlphaLessThanX = Less (AlphaCast, X)
   Y = Where (AlphaLessThanX, X, ZeroCast)
}
"""
```

ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.
