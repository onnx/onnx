```toml
name = "QLinearConv"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::QLinearConvShapeInference_opset10"

[attributes.auto_pad]
type = "STRING"
desc = "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`. The padding is split between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER."
default = "NOTSET"

[attributes.dilations]
type = "INTS"
desc = "dilation value along each spatial axis of the filter. If not present, the dilation defaults to 1 along each spatial axis."

[attributes.group]
type = "INT"
desc = "number of groups input channels and output channels are divided into. default is 1."
default = 1

[attributes.kernel_shape]
type = "INTS"
desc = "The shape of the convolution kernel. If not present, should be inferred from input 'w'."

[attributes.pads]
type = "INTS"
desc = "Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.The value represent the number of pixels added to the beginning and end part of the corresponding axis.`pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number ofpixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`.This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaultsto 0 along start and end of each spatial axis."

[attributes.strides]
type = "INTS"
desc = "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis."

[[inputs]]
name = "x"
type = "T1"
desc = "Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn). Optionally, if dimension denotation is in effect, the operation expects input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...]."

[[inputs]]
name = "x_scale"
type = "tensor(float)"
desc = "Scale tensor for input 'x'. It's a scalar, which means a per-tensor/layer quantization."

[[inputs]]
name = "x_zero_point"
type = "T1"
desc = "Zero point tensor for input 'x'. It's a scalar, which means a per-tensor/layer quantization."

[[inputs]]
name = "w"
type = "T2"
desc = "The weight tensor that will be used in the convolutions; has size (M x C/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the dimension of the kernel. Optionally, if dimension denotation is in effect, the operation expects the weight tensor to arrive with the dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. X.shape[1] == (W.shape[1] * group) == C (assuming zero based indices for the shape array). Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. "

[[inputs]]
name = "w_scale"
type = "tensor(float)"
desc = "Scale tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of output channels (M)."

[[inputs]]
name = "w_zero_point"
type = "T2"
desc = "Zero point tensor for input 'w'. It could be a scalar or a 1-D tensor, which means a per-tensor/layer or per output channel quantization. If it's a 1-D tensor, its number of elements should be equal to the number of output channels (M)."

[[inputs]]
name = "y_scale"
type = "tensor(float)"
desc = "Scale tensor for output 'y'. It's a scalar, which means a per-tensor/layer quantization."

[[inputs]]
name = "y_zero_point"
type = "T3"
desc = "Zero point tensor for output 'y'. It's a scalar, which means a per-tensor/layer quantization."

[[inputs]]
name = "B"
type = "T4"
desc = "Optional 1D bias to be added to the convolution, has size of M. Bias must be quantized using scale = x_scale * w_scale and zero_point = 0"
option = "optional"

[[outputs]]
name = "y"
type = "T3"
desc = "Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths."

[type_constraints.T1]
types = ["tensor(int8)", "tensor(uint8)"]
desc = "Constrain input type to 8-bit integer tensor."

[type_constraints.T2]
types = ["tensor(int8)", "tensor(uint8)"]
desc = "Constrain filter type to 8-bit integer tensor."

[type_constraints.T3]
types = ["tensor(int8)", "tensor(uint8)"]
desc = "Constrain output type to 8-bit integer tensor."

[type_constraints.T4]
types = ["tensor(int32)"]
desc = "Constrain bias type to 32-bit integer tensor."
```

The convolution operator consumes a quantized input tensor, its scale and zero point,
a quantized filter, its scale and zero point, and output's scale and zero point,
and computes the quantized output. Each scale and zero-point pair must have same shape.
It means they must be either scalars (per tensor) or 1-D tensors (per output channel).
Each input or output and its related zero point must have same type.
When bias is present it must be quantized using scale = input scale * weight scale and
zero point as 0.
