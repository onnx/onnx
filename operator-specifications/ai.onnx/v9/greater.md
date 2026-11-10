```toml
name = "Greater"
domain = "ai.onnx"
shape_inference = "defs::logical::utils::BinaryLogicShapeInference_opset12"

[type_constraints.T]
types = "all_numeric_types"
desc = "Constrain input types to all numeric tensors."

[type_constraints.T1]
types = ["tensor(bool)"]
desc = "Constrain output to boolean tensor."
```
