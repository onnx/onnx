```toml
name = "MaxUnpool"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::maxUnpoolShapeInference_opset9"

[attributes.kernel_shape]
type = "INTS"
desc = "The size of the kernel along each axis."
required = true

[attributes.pads]
type = "INTS"
desc = "Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis."

[attributes.strides]
type = "INTS"
desc = "Stride along each spatial axis."

[[inputs]]
name = "X"
type = "T1"
desc = "Input data tensor that has to be unpooled. This tensor is typically the first output of the MaxPool op.Dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non-image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...]."

[[inputs]]
name = "I"
type = "T2"
desc = "Input data tensor containing the indices corresponding to elements in the first input tensor X.This tensor is typically the second output of the MaxPool op.Dimensions must be the same as input tensor X. The indices are linear, i.e. computed considering the tensor as flattened 1-D tensor, assuming row-major storage. Also, the linear indices should not consider padding. So the values in indices are in the range [0, N x C x D1 x ... x Dn)."

[[inputs]]
name = "output_shape"
type = "T2"
desc = "The shape of the output can be explicitly set which will cause pads values to be auto generated. If 'output_shape' is specified, 'pads' values are ignored."
option = "optional"

[[outputs]]
name = "output"
type = "T1"
desc = "Output data tensor that contains the result of the unpooling."

[type_constraints.T1]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."

[type_constraints.T2]
types = ["tensor(int64)"]
desc = "Constrain index tensor to int64"
```

MaxUnpool essentially computes the partial inverse of the MaxPool op.
 The input information to this op is typically the output information from a MaxPool op. The first
 input tensor X is the tensor that needs to be unpooled, which is typically the pooled tensor (first output)
 from MaxPool. The second input tensor, I, contains the indices to the (locally maximal) elements corresponding
 to the elements in the first input tensor X. Input tensor I is typically the second output of the MaxPool op.
 The third (optional) input is a tensor that specifies the output size of the unpooling operation.

MaxUnpool is intended to do 'partial' inverse of the MaxPool op. 'Partial' because all the non-maximal
 values from the original input to MaxPool are set to zero in the output of the MaxUnpool op. Pooling
 the result of an unpooling operation should give back the original input to the unpooling op.

MaxUnpool can produce the same output size for several input sizes, which makes unpooling op ambiguous.
 The third input argument, output_size, is meant to disambiguate the op and produce output tensor of
 known/predictable size.

In addition to the inputs, MaxUnpool takes three attributes, namely kernel_shape, strides, and pads,
 which define the exact unpooling op. The attributes typically have the same values as the corresponding
 pooling op that the unpooling op is trying to invert.
