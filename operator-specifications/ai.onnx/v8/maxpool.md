```toml
name = "MaxPool"
domain = "ai.onnx"
shape_inference = "defs::nn::utils::PoolShapeInference_opset1_to_8"

[attributes.auto_pad]
type = "STRING"
desc = "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input.In case of odd number add the extra padding at the end for SAME_UPPER and at the beginning for SAME_LOWER. VALID mean no padding."
default = "NOTSET"

[attributes.kernel_shape]
type = "INTS"
desc = "The size of the kernel along each axis."
required = true

[attributes.pads]
type = "INTS"
desc = "Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represent the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults to 0 along start and end of each spatial axis."

[attributes.storage_order]
type = "INT"
desc = "The storage order of the tensor. 0 is row major, and 1 is column major."
default = 0

[attributes.strides]
type = "INTS"
desc = "Stride along each spatial axis."

[[outputs]]
name = "Y"
type = "T"
desc = "Output data tensor from average or max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used"

[[outputs]]
name = "Indices"
type = "I"
desc = "Indices tensor from max pooling across the input tensor. The dimensions of indices are the same as output tensor. The values in indices of are the indices of the selected values during pooling. The indices are computed as flatten 1-D tensor, and the indices do not consider padding. So the values in indices are in [0, N x C x D1 x ... x Dn)."
option = "optional"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."

[type_constraints.I]
types = ["tensor(int64)"]
desc = "Constrain index tensor to int64"
```
