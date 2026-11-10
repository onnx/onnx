```toml
name = "Min"
domain = "ai.onnx"
shape_inference = "defs::math::utils::ElementwiseMultiOpShapeInference"

[[inputs]]
name = "data_0"
type = "T"
desc = "List of tensors for min."
option = "variadic"
differentiation = "differentiable"

[[outputs]]
name = "min"
type = "T"
desc = "Output tensor."
differentiation = "differentiable"

[type_constraints.T]
types = "all_numeric_types_ir4"
desc = "Constrain input and output types to numeric tensors."
```
