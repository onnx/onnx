```toml
name = "Erf"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor"

[[outputs]]
name = "output"
type = "T"
desc = "The error function of the input tensor computed element-wise. It has the same shape and type of the input."

[type_constraints.T]
types = "all_numeric_types"
desc = "Constrain input and output types to all numeric tensors."
```

Computes the error function of the given input tensor element-wise.
