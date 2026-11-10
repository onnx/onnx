```toml
name = "CumProd"
domain = "ai.onnx"

[attributes.exclusive]
type = "INT"
desc = "If set to 1 will return exclusive product in which the top element is not included. In other terms, if set to 1, the j-th output element would be the product of the first (j-1) elements. Otherwise, it would be the product of the first j elements."
default = 0

[attributes.reverse]
type = "INT"
desc = "If set to 1 will perform the products in reverse direction."
default = 0

[[inputs]]
name = "x"
type = "T"
desc = "An input tensor that is to be processed."
differentiation = "differentiable"

[[inputs]]
name = "axis"
type = "T2"
desc = "A 0-D tensor. Must be in the range [-rank(x), rank(x)-1]. Negative value means counting dimensions from the back."
differentiation = "non_differentiable"

[[outputs]]
name = "y"
type = "T"
desc = "Output tensor of the same type as 'x' with cumulative products of the x's elements"
differentiation = "differentiable"

[type_constraints.T]
types = "numeric_types_for_math_reduction_ir4"
desc = "Constrain input and output types to numeric tensors."

[type_constraints.T2]
types = ["tensor(int32)", "tensor(int64)"]
desc = "axis tensor can be int32 or int64 only"
```

Performs cumulative product of the input elements along the given axis.
By default, it will do the product inclusively meaning the first element is copied as is.
Through an `exclusive` attribute, this behavior can change to exclude the first element.
It can also perform product in the opposite direction of the axis. For that, set `reverse` attribute to 1.

Example:
```
input_x = [1, 2, 3]
axis=0
output = [1, 2, 6]
exclusive=1
output = [1, 1, 2]
exclusive=0
reverse=1
output = [6, 6, 3]
exclusive=1
reverse=1
output = [6, 3, 1]
```
