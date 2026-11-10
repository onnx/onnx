```toml
name = "Conv"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::ConvShapeInference_opset1"

[attributes.auto_pad]
type = "STRING"
desc = "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input.In case of odd number add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding."
default = "NOTSET"

[attributes.dilations]
type = "INTS"
desc = "dilation value along each spatial axis of the filter."

[attributes.group]
type = "INT"
desc = "number of groups input channels and output channels are divided into."
default = 1

[attributes.kernel_shape]
type = "INTS"
desc = "The shape of the convolution kernel. If not present, should be inferred from input W."

[attributes.pads]
type = "INTS"
desc = "Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis."

[attributes.strides]
type = "INTS"
desc = "Stride along each spatial axis."

[[inputs]]
name = "X"
type = "T"
desc = "Input data tensor from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image. Otherwise the size is (N x C x D1 x D2 ... x Dn). Optionally, if dimension denotation is in effect, the operation expects input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...]."

[[inputs]]
name = "W"
type = "T"
desc = "The weight tensor that will be used in the convolutions; has size (M x C/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps. For more than 2 dimensions, the kernel shape will be (M x C/group x k1 x k2 x ... x kn), where (k1 x k2 x ... kn) is the dimension of the kernel. Optionally, if dimension denotation is in effect, the operation expects the weight tensor to arrive with the dimension denotation of [FILTER_OUT_CHANNEL, FILTER_IN_CHANNEL, FILTER_SPATIAL, FILTER_SPATIAL ...]. X.shape[1] == (W.shape[1] * group) == C (assuming zero based indices for the shape array). Or in other words FILTER_IN_CHANNEL should be equal to DATA_CHANNEL. "

[[inputs]]
name = "B"
type = "T"
desc = "Optional 1D bias to be added to the convolution, has size of M."
option = "optional"

[[outputs]]
name = "Y"
type = "T"
desc = "Output data tensor that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths."

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

The convolution operator consumes an input tensor and a filter, and
computes the output.
