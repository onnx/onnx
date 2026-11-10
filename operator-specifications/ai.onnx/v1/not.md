```toml
name = "Not"
domain = "ai.onnx"
shape_inference = "defs::logical::utils::unaryLogicalOpInference"

[[inputs]]
name = "X"
type = "T"
desc = "Input tensor"
differentiation = "non_differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "Output tensor"
differentiation = "non_differentiable"

[type_constraints.T]
types = ["tensor(bool)"]
desc = "Constrain input/output to boolean tensors."
```

Returns the negation of the input tensor element-wise.
