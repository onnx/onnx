```toml
name = "CastLike"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::CastLikeShapeInference_opset21"
function_body_builders = { 21 = "defs::tensor::utils::CastLikeFunctionBody_opset21" }

[type_constraints.T1]
types = "all_non_complex_tensor_types_ir10"
desc = "Constrain input types. Casting from complex is not supported."

[type_constraints.T2]
types = "all_non_complex_tensor_types_ir10"
desc = "Constrain output types. Casting to complex is not supported."
```
