```toml
name = "DepthToSpace"
domain = "ai.onnx"

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
desc = "Output tensor of [N, C/(blocksize * blocksize), H * blocksize, W * blocksize]."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to all tensor types."
```

DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions.
