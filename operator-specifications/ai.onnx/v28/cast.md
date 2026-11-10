```toml
name = "Cast"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::CastShapeInference_opset28"
data_propagation = "defs::tensor::utils::CastDataPropagation_opset28"

[type_constraints.T1]
types = "all_non_complex_tensor_types_ir14"
desc = "Constrain input types. Casting from complex is not supported."

[type_constraints.T2]
types = "all_non_complex_tensor_types_ir14"
desc = "Constrain output types. Casting to complex is not supported."
```
