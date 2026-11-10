```toml
name = "Einsum"
domain = "ai.onnx"
shape_inference = "defs::math::utils::EinsumShapeInference"

[type_constraints.T]
types = "all_numeric_types_ir4"
desc = "Constrain input and output types to all numerical tensor types."
```
