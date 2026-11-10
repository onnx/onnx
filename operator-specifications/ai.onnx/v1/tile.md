```toml
name = "Tile"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::TileShapeInference_opset1"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor of any shape."

[[inputs]]
name = "tiles"
type = "T"
desc = "Number of repeated copies to make of the input tensor."

[[inputs]]
name = "axis"
type = "T"
desc = "Axis along which to repeat."

[[outputs]]
name = "output"
type = "T"
desc = "Output tensor of same shape and type as input."

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input types to float tensors."

[type_constraints.T1]
types = ["tensor(int64)"]
desc = "Constrain tiles and axis's type to int64 tensors."
```

Repeat the elements of a tensor along an axis.
