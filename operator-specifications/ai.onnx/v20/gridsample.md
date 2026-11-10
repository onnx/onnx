```toml
name = "GridSample"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::gridSampleShapeInference"

[attributes.align_corners]
type = "INT"
desc = "If align_corners=1, the extrema (-1 and 1) are considered as referring to the center points of the input's corner pixels (voxels, etc.). If align_corners=0, they are instead considered as referring to the corner points of the input's corner pixels (voxels, etc.), making the sampling more resolution agnostic."
default = 0

[attributes.mode]
type = "STRING"
desc = "Three interpolation modes: linear (default), nearest and cubic. The \"linear\" mode includes linear and N-linear interpolation modes depending on the number of spatial dimensions of the input tensor (i.e. linear for 1 spatial dimension, bilinear for 2 spatial dimensions, etc.). The \"cubic\" mode also includes N-cubic interpolation modes following the same rules. The \"nearest\" mode rounds to the nearest even index when the sampling point falls halfway between two indices."
default = "linear"

[attributes.padding_mode]
type = "STRING"
desc = "Support padding modes for outside grid values: `zeros`(default), `border`, `reflection`. zeros: use 0 for out-of-bound grid locations, border: use border values for out-of-bound grid locations, reflection: use values at locations reflected by the border for out-of-bound grid locations. If index 0 represents the margin pixel, the reflected value at index -1 will be the same as the value at index 1. For location far away from the border, it will keep being reflected until becoming in bound. If pixel location x = -3.5 reflects by border -1 and becomes x' = 1.5, then reflects by border 1 and becomes x'' = 0.5."
default = "zeros"

[[inputs]]
name = "X"
type = "T1"
desc = "Input tensor of rank r+2 that has shape (N, C, D1, D2, ..., Dr), where N is the batch size, C is the number of channels, D1, D2, ..., Dr are the spatial dimensions."
differentiation = "differentiable"

[[inputs]]
name = "grid"
type = "T2"
desc = "Input offset of shape (N, D1_out, D2_out, ..., Dr_out, r), where D1_out, D2_out, ..., Dr_out are the spatial dimensions of the grid and output, and r is the number of spatial dimensions. Grid specifies the sampling locations normalized by the input spatial dimensions. Therefore, it should have most values in the range of [-1, 1]. If the grid has values outside the range of [-1, 1], the corresponding outputs will be handled as defined by padding_mode. Following computer vision convention, the coordinates in the length-r location vector are listed from the innermost tensor dimension to the outermost, the opposite of regular tensor indexing."
differentiation = "non_differentiable"

[[outputs]]
name = "Y"
type = "T1"
desc = "Output tensor of rank r+2 that has shape (N, C, D1_out, D2_out, ..., Dr_out) of the sampled values. For integer input types, intermediate values are computed as floating point and cast to integer at the end."
differentiation = "differentiable"
```

Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from the `grid`.
For spatial input `X` with shape (N, C, H, W), the `grid` will have shape (N, H_out, W_out, 2),
the output `Y` will have shape (N, C, H_out, W_out). For volumetric input `X` with shape (N, C, D, H, W),
the `grid` will have shape (N, D_out, H_out, W_out, 3), the output `Y` will have shape (N, C, D_out, H_out, W_out).
More generally, for an input `X` of rank r+2 with shape (N, C, d1, d2, ..., dr),
the `grid` will have shape (N, D1_out, D2_out, ..., Dr_out, r), the output `Y` will have shape (N, C, D1_out, D2_out, ..., Dr_out).

The tensor `X` contains values at centers of square pixels (voxels, etc) locations such as (n, c, d1_in, d2_in, ..., dr_in).
The (n, d1_out, d2_out, ..., dr_out, :) values from the tensor `grid` are the normalized positions for interpolating the values
at the (n, c, d1_out, d2_out, ..., dr_out) locations from the output tensor `Y` using a specified interpolation method (the mode)
and a padding mode (for `grid` positions falling outside the 2-dimensional image).

For example, the values in `grid[n, h_out, w_out, :]` are size-2 vectors specifying normalized positions in the 2-dimensional space of `X`.
They are used to interpolate output values of `Y[n, c, h_out, w_out]`.

The GridSample operator is often used in doing grid generator and sampler in the
[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).
See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).
