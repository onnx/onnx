```toml
name = "Hardmax"
domain = "ai.onnx"
shape_inference = "defs::math::utils::HardmaxShapeInference_opset13"

[attributes.axis]
type = "INT"
desc = """

Describes the dimension Hardmax will be performed on.
Negative value means counting dimensions
from the back. Accepted range is [-r, r-1] where r = rank(input).
"""
default = -1

[[inputs]]
name = "input"
type = "T"
desc = "The input tensor of rank >= axis."
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "The output values with the same shape as the input tensor."
differentiation = "differentiable"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."
```

The operator computes the hardmax values for the given input:

 Hardmax(element in input, axis) = 1 if the element is the first maximum value along the specified axis, 0 otherwise

The "axis" attribute indicates the dimension along which Hardmax
will be performed. The output tensor has the same shape
and contains the Hardmax values of the corresponding input.
