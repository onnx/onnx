```toml
name = "SpaceToDepth"
domain = "ai.onnx"

[attributes.blocksize]
type = "INT"
desc = "Blocks of [blocksize, blocksize] are moved."
required = true

[attributes.mode]
type = "STRING"
desc = "DCR (default) for depth-column-row order re-arrangement. Use CRD for column-row-depth order."
default = "DCR"
```

SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension. `mode` determines whether blocks are ordered depth-column-row
(`DCR`, the default) or column-row-depth (`CRD`).
