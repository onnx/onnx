```toml
name = "SpaceToDepth"
domain = "ai.onnx"

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width."
differentiation = "differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Output tensor of [N, C * blocksize * blocksize, H/blocksize, W/blocksize]."
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to all tensor types."
```
