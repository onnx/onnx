```toml
name = "SpaceToDepth"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::SpaceToDepthShapeInference_opset1"

[attributes.blocksize]
type = "INT"
desc = "Blocks of [blocksize, blocksize] are moved."
required = true

[[inputs]]
name = "input"
type = "T"
desc = "Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width."

[[outputs]]
name = "output"
type = "T"
desc = "Output tensor of [N, C * blocksize * blocksize, H/blocksize, W/blocksize]."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to all tensor types."
```

SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.
