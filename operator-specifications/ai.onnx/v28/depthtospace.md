```toml
name = "DepthToSpace"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::DepthToSpaceShapeInference_opset28"
function_body_builders = { 28 = "defs::tensor::utils::BuildContextDependentFunctionBodyDepthToSpace" }
```
