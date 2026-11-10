```toml
name = "LogSoftmax"
domain = "ai.onnx"
node_determinism = "deterministic"

[attributes.axis]
type = "INT"
desc = """

Describes the dimension LogSoftmax will be performed on.
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

The operator computes the log of softmax values for the given input:

 LogSoftmax(input, axis) = Log(Softmax(input, axis=axis))

The "axis" attribute indicates the dimension along which LogSoftmax
will be performed. The output tensor has the same shape
and contains the LogSoftmax values of the corresponding input.
