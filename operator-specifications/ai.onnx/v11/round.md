```toml
name = "Round"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[[inputs]]
name = "X"
type = "T"
desc = "Input tensor"
differentiation = "non_differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor"
differentiation = "non_differentiable"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Round takes one input Tensor and rounds the values, element-wise, meaning
it finds the nearest integer for each value.
In case of halves, the rule is to round them to the nearest even integer.
If input x is integral, +0, -0, NaN,  or infinite, x itself is returned.
The output tensor has the same shape and type as the input.

Examples:
```
round([0.9]) = [1.0]
round([2.5]) = [2.0]
round([2.3]) = [2.0]
round([1.5]) = [2.0]
round([-4.5]) = [-4.0]
```
