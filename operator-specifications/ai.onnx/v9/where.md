```toml
name = "Where"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::WhereShapeInference_opset9"

[[inputs]]
name = "condition"
type = "B"
desc = "When True (nonzero), yield X, otherwise yield Y"
differentiation = "non_differentiable"

[[inputs]]
name = "X"
type = "T"
desc = "values selected at indices where condition is True"
differentiation = "differentiable"

[[inputs]]
name = "Y"
type = "T"
desc = "values selected at indices where condition is False"
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Tensor of shape equal to the broadcasted shape of condition, X, and Y."
differentiation = "differentiable"

[type_constraints.B]
types = ["tensor(bool)"]
desc = "Constrain to boolean tensors."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to all tensor types."
```

Return elements, either from X or Y, depending on condition.
Where behaves like
[numpy.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)
with three parameters.

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).
