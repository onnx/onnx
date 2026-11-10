```toml
name = "Add"
domain = "ai.onnx"
shape_inference = "propagateShapeAndTypeFromFirstInput"

[attributes.axis]
type = "INT"
desc = "If set, defines the broadcast dimensions. See doc for details."

[attributes.broadcast]
type = "INT"
desc = "Pass 1 to enable broadcasting"
default = 0

[type_constraints.T]
types = "numeric_types_for_math_reduction"
desc = "Constrain input and output types to high-precision numeric tensors."
```
