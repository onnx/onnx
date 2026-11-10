```toml
name = "Gemm"
domain = "ai.onnx"
shape_inference = "defs::math::utils::GemmShapeInference_opset6"

[[inputs]]
name = "A"
type = "T"
desc = "Input tensor A"

[[inputs]]
name = "B"
type = "T"
desc = "Input tensor B"

[[inputs]]
name = "C"
type = "T"
desc = "Input tensor C"
```
