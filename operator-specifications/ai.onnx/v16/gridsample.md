```toml
name = "GridSample"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::gridSampleShapeInference"

[attributes.align_corners]
type = "INT"
desc = "If align_corners=1, the extrema (-1 and 1) are considered as referring to the center points of the input's corner pixels. If align_corners=0, they are instead considered as referring to the corner points of the input's corner pixels, making the sampling more resolution agnostic."
default = 0

[attributes.mode]
type = "STRING"
desc = "Three interpolation modes: bilinear (default), nearest and bicubic."
default = "bilinear"

[attributes.padding_mode]
type = "STRING"
desc = "Support padding modes for outside grid values: `zeros`(default), `border`, `reflection`. zeros: use 0 for out-of-bound grid locations, border: use border values for out-of-bound grid locations, reflection: use values at locations reflected by the border for out-of-bound grid locations. If index 0 represents the margin pixel, the reflected value at index -1 will be the same as the value at index 1. For location far away from the border, it will keep being reflected until becoming in bound. If pixel location x = -3.5 reflects by border -1 and becomes x' = 1.5, then reflects by border 1 and becomes x'' = 0.5."
default = "zeros"

[[inputs]]
name = "X"
type = "T1"
desc = "4-D tensor of shape (N, C, H, W), where N is the batch size, C is the numbers of channels, H and W are the height and width of the input data."
differentiation = "differentiable"

[[inputs]]
name = "grid"
type = "T2"
desc = "Input offset, 4-D tensor of shape (N, H_out, W_out, 2), where H_out and W_out are the height and width of grid and output, Grid specifies the sampling pixel locations normalized by the input spatial dimensions. Therefore, it should have most values in the range of [-1, 1]. If grid has values outside the range of [-1, 1], the corresponding outputs will be handled as defined by padding_mode."
differentiation = "non_differentiable"

[[outputs]]
name = "Y"
type = "T1"
desc = "4-D tensor of shape (N, C, H_out, W_out) of sampled values. For integer input types, intermediate values are computed as floating point and cast to integer at the end."
differentiation = "differentiable"

[type_constraints.T1]
types = "all_tensor_types"
desc = "Constrain input `X` and output `Y` types to all tensor types."

[type_constraints.T2]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain grid types to float tensors."
```

Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from `grid`.
Currently, only spatial (4-D) inputs are supported. For input `X` with shape (N, C, H, W) and `grid` with shape (N, H_out, W_out, 2),
the output `Y` will have shape (N, C, H_out, W_out).

The tensor `X` contains values at centers of square pixels in a H by W 2-dimensional image.
The tensor `grid` describes normalized positions where the output `Y` is to be computed
using a specified interpolation method (the mode) and a padding mode (for grid positions falling outside the 2-dimensional image).

Elements in `grid[N, H_out, W_out]` are size-2 vectors specifying positions in the 2-dimensional space of `X`.
They are used to interpolate output values of `Y[N, C, H_out, W_out]`.

The GridSample operator is often used in doing grid generator and sampler in the [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).
See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/master/generated/torch.nn.functional.grid_sample.html#torch-nn-functional-grid-sample).
