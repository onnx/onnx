```toml
name = "Concat"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::ConcatShapeInference_opset4"

[attributes.axis]
type = "INT"
desc = "Which axis to concat on"
required = true

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain output types to any tensor type."
```
