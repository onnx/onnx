```toml
name = "Shrink"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[attributes.bias]
type = "FLOAT"
desc = "The bias value added to output. Default is 0."
default = 0.0

[attributes.lambd]
type = "FLOAT"
desc = "The lambd value for the Shrink formulation. Default is 0.5."
default = 0.5

[[inputs]]
name = "input"
type = "T"
desc = "The input data as Tensor."
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "The output."
differentiation = "differentiable"

[type_constraints.T]
types = "all_numeric_types"
desc = "Constrain input to only numeric types."

[function_bodies]
18 = """
{
   Lambd = Constant <value_float: float = @lambd> ()
   LambdCast = CastLike (Lambd, input)
   Bias = Constant <value_float: float = @bias> ()
   BiasCast = CastLike (Bias, input)
   Zero = Constant <value: tensor = float {0}> ()
   ZeroCast = CastLike (Zero, input)
   NegLmbda = Neg (LambdCast)
   InputLessThanNegLambda = Less (input, NegLmbda)
   InputAddBias = Add (input, BiasCast)
   InputSubBias = Sub (input, BiasCast)
   LambdaLessThanInput = Less (LambdCast, input)
   InputSubBiasOrZero = Where (LambdaLessThanInput, InputSubBias, ZeroCast)
   output = Where (InputLessThanNegLambda, InputAddBias, InputSubBiasOrZero)
}
"""
```

Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
having same datatype and shape with input. It has two attributes, lambd and
bias. The formula of this operator is: If x < -lambd, y = x + bias;
If x > lambd, y = x - bias; Otherwise, y = 0.
