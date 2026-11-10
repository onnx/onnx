```toml
name = "Floor"
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
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."
```

Floor takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the floor is, y = floor(x), is applied to
the tensor elementwise. If x is integral, +0, -0, NaN,  or infinite, x itself is returned.
