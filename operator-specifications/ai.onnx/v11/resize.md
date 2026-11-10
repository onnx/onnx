```toml
name = "Resize"
domain = "ai.onnx"

[attributes.coordinate_transformation_mode]
type = "STRING"
desc = """

This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor. <br/>

The coordinate of each dimension is transformed individually. Let's describe a case using axis x as an example.
Denote x_resized as the coordinate of axis x in the resized tensor, x_original as the coordinate of axis x in the original tensor, length_original as the length of the original tensor in axis x, length_resized as the length of the resized tensor in axis x, roi_x = (start_x, end_x) of the axis x in input \"roi\", scale = length_resized / length_original, <br/>

if coordinate_transformation_mode is \"half_pixel\", <br/>
x_original = (x_resized + 0.5) / scale - 0.5, <br/>

if coordinate_transformation_mode is \"pytorch_half_pixel\", <br/>
x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0, <br/>

if coordinate_transformation_mode is \"align_corners\", <br/>
x_original = x_resized * (length_original - 1) / (length_resized - 1), <br/>

if coordinate_transformation_mode is \"asymmetric\", <br/>
x_original = x_resized / scale, <br/>

if coordinate_transformation_mode is \"tf_half_pixel_for_nn\", <br/>
x_original = (x_resized + 0.5) / scale, <br/>

if coordinate_transformation_mode is \"tf_crop_and_resize\", <br/>
x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1)."""
default = "half_pixel"

[attributes.cubic_coeff_a]
type = "FLOAT"
desc = "The coefficient 'a' used in cubic interpolation. Two common choice are -0.5 (in some cases of TensorFlow) and -0.75 (in PyTorch). Check out Equation (4) in https://ieeexplore.ieee.org/document/1163711 for the details. This attribute is valid only if \"mode\" is \"cubic\"."
default = -0.75

[attributes.exclude_outside]
type = "INT"
desc = "If set to 1, the weight of sampling locations outside the tensor will be set to 0 and the weight will be renormalized so that their sum is 1.0. The default value is 0."
default = 0

[attributes.extrapolation_value]
type = "FLOAT"
desc = "When coordinate_transformation_mode is \"tf_crop_and_resize\" and x_original is outside the range [0, length_original - 1], this value is used as the corresponding output value. Default is 0.0f."
default = 0.0

[attributes.mode]
type = "STRING"
desc = "Three interpolation modes: nearest (default), linear and cubic. The \"linear\" mode includes linear interpolation for 1D tensor and N-linear interpolation for N-D tensor (for example, bilinear interpolation for 2D tensor). The \"cubic\" mode includes cubic interpolation for 1D tensor and N-cubic interpolation for N-D tensor (for example, bicubic interpolation for 2D tensor)."
default = "nearest"

[attributes.nearest_mode]
type = "STRING"
desc = "Four modes: round_prefer_floor (default, as known as round half down), round_prefer_ceil (as known as round half up), floor, ceil. Only used by nearest interpolation. It indicates how to get \"nearest\" pixel in input tensor from x_original, so this attribute is valid only if \"mode\" is \"nearest\"."
default = "round_prefer_floor"

[[inputs]]
name = "X"
type = "T1"
desc = "N-D tensor"

[[inputs]]
name = "roi"
type = "T2"
desc = "1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of X. The RoIs' coordinates are normalized in the coordinate system of the input image. It only takes effect when coordinate_transformation_mode is \"tf_crop_and_resize\""

[[inputs]]
name = "scales"
type = "tensor(float)"
desc = "The scale array along each dimension. It takes value greater than 0. If it's less than 1, it's sampling down, otherwise, it's upsampling. The number of elements of 'scales' should be the same as the rank of input 'X'. If 'size' is needed, the user must set 'scales' to an empty tensor."

[[inputs]]
name = "sizes"
type = "tensor(int64)"
desc = "The size of the output tensor. The number of elements of 'sizes' should be the same as the rank of input 'X'. May only be set if 'scales' is set to an empty tensor."
option = "optional"

[[outputs]]
name = "Y"
type = "T1"
desc = "N-D tensor after resizing"

[type_constraints.T1]
types = "all_tensor_types"
desc = "Constrain input 'X' and output 'Y' to all tensor types."

[type_constraints.T2]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain roi type to float or double."
```

Resize the input tensor. In general, it calculates every value in the output tensor as a weighted average of neighborhood (a.k.a. sampling locations) in the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) if input \"sizes\" is not specified.
