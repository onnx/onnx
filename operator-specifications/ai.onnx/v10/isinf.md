```toml
name = "IsInf"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::IsInfShapeInference_opset10"

[attributes.detect_negative]
type = "INT"
desc = "(Optional) Whether map negative infinity to true. Default to 1 so that negative infinity induces true. Set this attribute to 0 if negative infinity should be mapped to false."
default = 1

[attributes.detect_positive]
type = "INT"
desc = "(Optional) Whether map positive infinity to true. Default to 1 so that positive infinity induces true. Set this attribute to 0 if positive infinity should be mapped to false."
default = 1

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
types = ["tensor(float)", "tensor(double)"]
desc = "Constrain input types to float tensors."

[type_constraints.T2]
types = ["tensor(bool)"]
desc = "Constrain output types to boolean tensors."
```

Map infinity to true and other values to false.
