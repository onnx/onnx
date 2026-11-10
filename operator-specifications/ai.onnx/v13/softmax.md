```toml
name = "Softmax"
domain = "ai.onnx"
node_determinism = "deterministic"
shape_inference = "defs::math::utils::SoftmaxShapeInference_opset13"
# function body builder for opset version 13 (the default opset version is the same
# as the operator's since_version.
# function body builder for opset version 18.
# ReduceSum is updated in opset 18 to have axes as the second input.
# Therefore function body for opset version 18
# is different than the one defined using opset version 13.
function_body_builders = { 13 = "defs::math::utils::SoftmaxFunctionBody_opset13", 18 = "defs::math::utils::SoftmaxFunctionBody_opset18" }

[attributes.axis]
type = "INT"
desc = """

Describes the dimension Softmax will be performed on.
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

The operator computes the normalized exponential values for the given input:

 Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)

The "axis" attribute indicates the dimension along which Softmax
will be performed. The output tensor has the same shape
and contains the Softmax values of the corresponding input.
