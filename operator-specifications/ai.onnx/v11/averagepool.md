```toml
name = "AveragePool"
domain = "ai.onnx"

[attributes.auto_pad]
type = "STRING"
desc = "auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the input so that `output_shape[i] = ceil(input_shape[i] / strides[i])` for each axis `i`. The padding is split between the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER."
default = "NOTSET"

[attributes.ceil_mode]
type = "INT"
desc = "Whether to use ceil or floor (default) to compute the output shape."
default = 0

[attributes.count_include_pad]
type = "INT"
desc = "Whether include pad pixels when calculating values for the edges. Default is 0, doesn't count include pad."
default = 0

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
type = "T"
desc = "Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size. Optionally, if dimension denotation is in effect, the operation expects the input data tensor to arrive with the dimension denotation of [DATA_BATCH, DATA_CHANNEL, DATA_FEATURE, DATA_FEATURE ...]."
differentiation = "differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "Output data tensor from average or max pooling across the input tensor. Dimensions will vary based on various kernel, stride, and pad sizes. Floor value of the dimension is used"
differentiation = "differentiable"
```

AveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled

 ```
 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
or when ceil_mode is disabled:
 ```
 VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor(input_spatial_shape[i] / strides_spatial_shape[i])
 ```

 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
 ```
 The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
