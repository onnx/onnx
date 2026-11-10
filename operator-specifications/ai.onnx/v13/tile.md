```toml
name = "Tile"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::TileShapeInference_opset13"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor of any shape."
differentiation = "differentiable"

[[inputs]]
name = "repeats"
type = "T1"
desc = "1D int64 tensor of the same length as input's dimension number, includes numbers of repeated copies along input's dimensions."
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Output tensor of the same dimensions and type as tensor input. output_dim[i] = input_dim[i] * repeats[i]"
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to all tensor types."

[type_constraints.T1]
types = ["tensor(int64)"]
desc = "Constrain repeat's type to int64 tensors."
```
