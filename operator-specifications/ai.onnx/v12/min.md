```toml
name = "Min"
domain = "ai.onnx"
shape_inference = "defs::math::utils::ElementwiseMultiOpShapeInference_opset8"

[type_constraints.T]
types = "all_numeric_types"
desc = "Constrain input and output types to numeric tensors."
```
