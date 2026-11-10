```toml
name = "Mean"
domain = "ai.onnx"
shape_inference = "defs::math::utils::ElementwiseMultiOpShapeInference"

[[inputs]]
name = "data_0"
type = "T"
desc = "List of tensors for mean."
option = "variadic"
differentiation = "differentiable"

[[outputs]]
name = "mean"
type = "T"
desc = "Output tensor."
differentiation = "differentiable"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."
```
