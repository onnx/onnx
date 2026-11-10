```toml
name = "ReverseSequence"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::ReverseSequenceShapeInference_opset28"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Input and output types can be of any tensor type."
```
