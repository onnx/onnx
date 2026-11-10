```toml
name = "Sign"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor"

[[outputs]]
name = "output"
type = "T"
desc = "The sign of the input tensor computed element-wise. It has the same shape and type of the input."

[type_constraints.T]
types = "all_numeric_types"
desc = "Constrain input and output types to all numeric tensors."
```

Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.
