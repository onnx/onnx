```toml
name = "OneHot"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::OneHotShapeInference_opset11"

[attributes.axis]
type = "INT"
desc = "(Optional) Axis along which one-hot representation in added. Default: axis=-1. axis=-1 means that the additional dimension will be inserted as the innermost/last dimension in the output tensor. Negative value means counting dimensions from the back. Accepted range is [-r-1, r] where r = rank(indices)."
default = -1

[[inputs]]
name = "indices"
type = "T1"
desc = "Input tensor containing indices. Any entries in the 'indices' input tensor with values outside the range [-depth, depth-1] will result in one-hot representation with all 'off_value' values in the output tensor.In case 'indices' is of non-integer type, the values will be casted to int64 before use."
differentiation = "non_differentiable"

[[inputs]]
name = "depth"
type = "T2"
desc = "Scalar or Rank 1 tensor containing exactly one element, specifying the number of classes in one-hot tensor. This is also the size of the one-hot dimension (specified by 'axis' attribute) added on in the output tensor. The values in the 'indices' input tensor are expected to be in the range [-depth, depth-1]. In case 'depth' is of non-integer type, it will be casted to int64 before use."
differentiation = "non_differentiable"

[[inputs]]
name = "values"
type = "T3"
desc = "Rank 1 tensor containing exactly two elements, in the format [off_value, on_value], where 'on_value' is the value used for filling locations specified in 'indices' input tensor, and 'off_value' is the value used for filling locations other than those specified in 'indices' input tensor. "
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T3"
desc = "Tensor of rank one greater than input tensor 'indices', i.e. rank(output) = rank(indices) + 1. The data type for the elements of the output tensor is the same as the type of input 'values' is used."
differentiation = "non_differentiable"
```

Produces a one-hot tensor based on inputs.
The locations represented by the index values in the 'indices' input tensor will have 'on_value'
and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
are specified as part of required input argument 'values', which is a two-element tensor of format
[off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
input tensor. The additional dimension is for one-hot representation. The additional dimension will
be inserted at the position specified by 'axis'. If 'axis' is not specified then the additional
dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
the range [-depth, depth-1] will result in one-hot representation with all 'off_value' values in the
output tensor.

when axis = 0:
output[input[i, j, k], i, j, k] = 1 for all i, j, k and 0 otherwise.

when axis = -1:
output[i, j, k, input[i, j, k]] = 1 for all i, j, k and 0 otherwise.
