```toml
name = "Identity"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor"

[[outputs]]
name = "output"
type = "T"
desc = "Tensor to copy input into."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to all tensor types."
```

Identity operator
