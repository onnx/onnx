```toml
name = "IsNaN"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::IsNaNShapeInference_opset13"

[[inputs]]
name = "X"
type = "T1"
desc = "input"
differentiation = "non_differentiable"

[[outputs]]
name = "Y"
type = "T2"
desc = "output"
differentiation = "non_differentiable"

[type_constraints.T1]
types = "all_float_types_ir4"
desc = "Constrain input types to float tensors."

[type_constraints.T2]
types = ["tensor(bool)"]
desc = "Constrain output types to boolean tensors."
```
