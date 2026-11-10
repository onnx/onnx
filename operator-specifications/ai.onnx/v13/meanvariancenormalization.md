```toml
name = "MeanVarianceNormalization"
domain = "ai.onnx"
shape_inference = false

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
desc = "Constrain input and output types to all numeric tensors."

[function_bodies]
13 = """
{
   Exponent = Constant <value: tensor = float {2}> ()
   Epsilon = Constant <value: tensor = float {1e-09}> ()
   X_RM = ReduceMean <axes: ints = @axes> (X)
   EX_squared = Pow (X_RM, Exponent)
   X_squared = Pow (X, Exponent)
   E_Xsquared = ReduceMean <axes: ints = @axes> (X_squared)
   Variance = Sub (E_Xsquared, EX_squared)
   STD = Sqrt (Variance)
   X_variance = Sub (X, X_RM)
   Processed_STD = Add (STD, Epsilon)
   Y = Div (X_variance, Processed_STD)
}
"""
18 = """
{
   Exponent = Constant <value: tensor = float {2}> ()
   Epsilon = Constant <value: tensor = float {1e-09}> ()
   axes = Constant <value_ints: ints = @axes> ()
   X_RM = ReduceMean (X, axes)
   EX_squared = Pow (X_RM, Exponent)
   X_squared = Pow (X, Exponent)
   E_Xsquared = ReduceMean (X_squared, axes)
   Variance = Sub (E_Xsquared, EX_squared)
   STD = Sqrt (Variance)
   X_variance = Sub (X, X_RM)
   Processed_STD = Add (STD, Epsilon)
   Y = Div (X_variance, Processed_STD)
}
"""
```

A MeanVarianceNormalization Function: Perform mean variance normalization
on the input tensor X using formula: `(X-EX)/sqrt(E(X-EX)^2)`
