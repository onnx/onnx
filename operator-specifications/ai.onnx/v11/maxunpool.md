```toml
name = "MaxUnpool"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::maxUnpoolShapeInference_opset11"

[attributes.kernel_shape]
type = "INTS"
desc = "The size of the kernel along each axis."
required = true

[attributes.pads]
type = "INTS"
desc = "Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis."

[attributes.strides]
type = "INTS"
desc = "Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis."

[[inputs]]
name = "X"
type = "T1"
desc = "Input data tensor that has to be unpooled. This tensor is typically the first output of the MaxPool op.Dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non-image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...]."
differentiation = "differentiable"

[[inputs]]
name = "I"
type = "T2"
desc = "Input data tensor containing the indices corresponding to elements in the first input tensor X.This tensor is typically the second output of the MaxPool op.Dimensions must be the same as input tensor X. The indices are linear, i.e. computed considering the tensor as flattened 1-D tensor, assuming row-major storage. Also, the linear indices should not consider padding. So the values in indices are in the range [0, N x C x D1 x ... x Dn)."
differentiation = "non_differentiable"

[[inputs]]
name = "output_shape"
type = "T2"
desc = "The shape of the output can be explicitly set which will cause pads values to be auto generated. If 'output_shape' is specified, 'pads' values are ignored."
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T1"
desc = "Output data tensor that contains the result of the unpooling."
differentiation = "differentiable"
```
