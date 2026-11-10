```toml
name = "DeformConv"
domain = "ai.onnx"

[attributes.dilations]
type = "INTS"
desc = "Dilation value along each spatial axis of the kernel. Default is 1 along each axis."

[attributes.group]
type = "INT"
desc = "Number of groups the input and output channels, C and oC, are divided into. C and oC must both be divisible by group. Default is 1."
default = 1

[attributes.kernel_shape]
type = "INTS"
desc = "Shape of the convolution kernel. If not present, it is inferred from the shape of input W."

[attributes.offset_group]
type = "INT"
desc = "Number of groups of offset. C must be divisible by offset_group. Default is 1."
default = 1

[attributes.pads]
type = "INTS"
desc = "Padding for the beginning and end along each spatial axis. The values represent the number of pixels added to the beginning and end of the corresponding axis and can take any nonnegative value. The format should be as follows: [x1_begin, x2_begin, ..., x1_end, x2_end, ...], where xi_begin is the number of pixels added at the beginning of axis `i` and xi_end is the number of pixels added at the end of axis `i`. Default is 0 along each axis."

[attributes.strides]
type = "INTS"
desc = "Stride along each spatial axis. Default is 1 along each axis."

[[inputs]]
name = "X"
type = "T"
desc = "Input data tensor. For 2D image data, it has shape (N, C, H, W) where N is the batch size, C is the number of input channels, and H and W are the height and width. In general, the shape is (N, C, D1, D2, ... , Dn) for n-dimensional data, where D1 to Dn are the spatial dimension sizes. Most common use cases have n = 2 or 3."

[[inputs]]
name = "W"
type = "T"
desc = "Weight tensor that will be used in the convolutions. It has shape (oC, C/group, kH, kW), where oC is the number of output channels and kH and kW are the kernel height and width. For more than 2 dimensions, it has shape (oC, C/group, k1, k2, ... , kn)."

[[inputs]]
name = "offset"
type = "T"
desc = "Offset tensor denoting the offset for the sampling locations in the convolution kernel. It has shape (N, offset_group * kH * kW * 2, oH, oW) for 2D data or (N, offset_group * k1 * k2 * ... * kn * n, o1, o2, ... , on) for nD data. Use linear interpolationfor fractional offset values. Sampling locations outside of the padded input tensor gives zero."

[[inputs]]
name = "B"
type = "T"
desc = "Optional 1D bias of length oC to be added to the convolution. Default is a tensor of zeros."
option = "optional"

[[inputs]]
name = "mask"
type = "T"
desc = "The mask tensor to be applied to each position in the convolution kernel. It has shape (N, offset_group * kH * kW, oH, oW) for 2D data or (N, offset_group * k1 * k2 * ... * kn * n, o1, o2, ... , on) for nD data. Default is a tensor of ones."
option = "optional"

[[outputs]]
name = "Y"
type = "T"
desc = "Output data tensor that contains the result of convolution. It has shape (N, oC, oH, oW) for 2D data or (N, oC, o1, o2, ..., on) for nD data"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Performs deformable convolution as described in https://arxiv.org/abs/1703.06211 and https://arxiv.org/abs/1811.11168.
This operator specification supports the general N-D case. Note that most common use cases have 2D or 3D data.
