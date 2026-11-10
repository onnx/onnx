````toml
name = "Resize"
domain = "ai.onnx"

[attributes.antialias]
type = "INT"
desc = "If set to 1, \"linear\" and \"cubic\" interpolation modes will use an antialiasing filter when downscaling. Antialiasing is achieved by stretching the resampling filter by a factor max(1, 1 / scale), which means that when downsampling, more input pixels contribute to an output pixel."
default = 0

[attributes.axes]
type = "INTS"
desc = "If provided, it specifies a subset of axes that 'roi', 'scales' and 'sizes' refer to. If not provided, all axes are assumed [0, 1, ..., r-1], where r = rank(data). Non-specified dimensions are interpreted as non-resizable. Negative value means counting dimensions from the back. Accepted range is [-r, r-1], where r = rank(data). Behavior is undefined if an axis is repeated."

[attributes.coordinate_transformation_mode]
type = "STRING"
desc = """

This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor.

The coordinate of each dimension is transformed individually. Let's describe a case using axis x as an example.
Denote `x_resized` as the coordinate of axis x in the resized tensor,
 `x_original` as the coordinate of axis x in the original tensor,
 `length_original` as the length of the original tensor in axis x,
 `length_resized` as the length of the resized tensor in axis x,
 `scale = length_resized / length_original`,
 `output_width` the target length on the axis x which can be a fractional number when it is calculated out of a scale factor,
 and `output_width_int` the effective output width as an integer.

if coordinate_transformation_mode is `\"half_pixel\"`,
```
x_original = (x_resized + 0.5) / scale - 0.5
```

if coordinate_transformation_mode is `\"half_pixel_symmetric\"`,
```
adjustment = output_width_int / output_width
center = input_width / 2
offset = center * (1 - adjustment)
x_ori = offset + (x + 0.5) / scale - 0.5
```

if coordinate_transformation_mode is `\"pytorch_half_pixel\"`,
```
x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0
```

if coordinate_transformation_mode is `\"align_corners\"`,
```
x_original = x_resized * (length_original - 1) / (length_resized - 1)
```

if coordinate_transformation_mode is `\"asymmetric\"`,
```
x_original = x_resized / scale
```

if coordinate_transformation_mode is `\"tf_crop_and_resize\"`,
```
x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1)
```
."""
default = "half_pixel"

[attributes.cubic_coeff_a]
type = "FLOAT"
desc = "The coefficient 'a' used in cubic interpolation. Two common choice are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch). Check out Equation (4) in https://ieeexplore.ieee.org/document/1163711 for the details. This attribute is valid only if mode is \"cubic\"."
default = -0.75

[attributes.exclude_outside]
type = "INT"
desc = "If set to 1, the weight of sampling locations outside the tensor will be set to 0 and the weight will be renormalized so that their sum is 1.0. The default value is 0."
default = 0

[attributes.extrapolation_value]
type = "FLOAT"
desc = "When coordinate_transformation_mode is \"tf_crop_and_resize\" and x_original is outside the range [0, length_original - 1], this value is used as the corresponding output value. Default is 0.0f."
default = 0.0

[attributes.keep_aspect_ratio_policy]
type = "STRING"
desc = """

This attribute describes how to interpret the `sizes` input with regard to keeping the original aspect ratio of the input, and it is not applicable when
the `scales` input is used.

Given a set of `sizes`, associated with a subset of `axes` (explicitly provided or default), and assuming `d = axes[i]`, with `i` being the index of the provided `sizes`.

If `keep_aspect_ratio_policy` is `\"stretch\"`, the original aspect ratio is disregarded, and the input is resized to the specified size:
`out_size[d] = sizes[i]`

If `keep_aspect_ratio_policy` is `\"not_larger\"`, the sizes are adjusted so that no extent of the output is larger than the specified size, while keeping the original aspect ratio:
```
scale = Min(sizes[i] / in_size[d])
out_size[d] = round_int(scale * in_size[d])
```

If `keep_aspect_ratio_policy` is `\"not_smaller\"`, the sizes are adjusted so that no extent of the output is smaller than the specified size, while keeping the original aspect ratio:
```
scale = Max(sizes[i] / in_size[d])
out_size[d] = round_int(scale * in_size[d])
```

For non-resizable axes (those not specified in `axes`), the output size will be equal to the input size.

Note: `round_int` stands for computing the nearest integer value, rounding halfway cases up."""
default = "stretch"

[attributes.mode]
type = "STRING"
desc = "Three interpolation modes: \"nearest\" (default), \"linear\" and \"cubic\". The \"linear\" mode includes linear interpolation for 1D tensor and N-linear interpolation for N-D tensor (for example, bilinear interpolation for 2D tensor). The \"cubic\" mode includes cubic interpolation for 1D tensor and N-cubic interpolation for N-D tensor (for example, bicubic interpolation for 2D tensor)."
default = "nearest"

[attributes.nearest_mode]
type = "STRING"
desc = "Four modes: \"round_prefer_floor\" (default, as known as round half down), \"round_prefer_ceil\" (as known as round half up), \"floor\", \"ceil\". Only used by nearest interpolation. It indicates how to get \"nearest\" pixel in input tensor from x_original, so this attribute is valid only if \"mode\" is \"nearest\"."
default = "round_prefer_floor"
````

Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
```
output_dimension = floor(input_dimension * (roi_end - roi_start) * scale)
```
if input \"sizes\" is not specified.
