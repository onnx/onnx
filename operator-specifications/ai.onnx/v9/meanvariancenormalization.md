```toml
name = "MeanVarianceNormalization"
domain = "ai.onnx"
shape_inference = false

[attributes.axes]
type = "INTS"
desc = "A list of integers, along which to reduce. The default is to calculate along axes [0,2,3] for calculating mean and variance along each channel. Two variables with the same C-coordinate are associated with the same mean and variance."
default = [0, 2, 3]

[[inputs]]
name = "X"
type = "T"
desc = "Input tensor"

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to all numeric tensors."

[function_bodies]
9 = """
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
```

A MeanVarianceNormalization Function: Perform mean variance normalization
on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ```
