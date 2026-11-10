```toml
name = "DeformConv"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::DeformConvShapeInference_opset22"

[type_constraints.T]
types = "all_float_types_ir4"
desc = "Constrain input and output types to float tensors."
```
