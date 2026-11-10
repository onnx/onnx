```toml
name = "Constant"
domain = "ai.onnx"
shape_inference = "defs::generator::utils::ConstantOpInference"

[type_constraints.T]
types = "all_tensor_types_ir12"
desc = "Constrain input and output types to all tensor types."
```
