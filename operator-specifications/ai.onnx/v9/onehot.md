```toml
name = "OneHot"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "(Optional) Axis along which one-hot representation in added. Default: axis=-1. axis=-1 means that the additional dimension will be inserted as the innermost/last dimension in the output tensor."
default = -1

[[inputs]]
name = "indices"
type = "T1"
desc = "Input tensor containing indices. The values must be non-negative integers. Any entries in the 'indices' input tensor with values outside the range [0, depth) will result in one-hot representation with all 'off_value' values in the output tensor.In case 'indices' is of non-integer type, the values will be casted to int64 before use."

[[inputs]]
name = "depth"
type = "T2"
desc = "Scalar or rank 1 tensor containing exactly one element, specifying the number of classes in one-hot tensor. This is also the size of the one-hot dimension (specified by 'axis' attribute) added on in the output tensor. The values in the 'indices' input tensor are expected to be in the range [0, depth). In case 'depth' is of non-integer type, it will be casted to int64 before use."

[[inputs]]
name = "values"
type = "T3"
desc = "Rank 1 tensor containing exactly two elements, in the format [off_value, on_value], where 'on_value' is the value used for filling locations specified in 'indices' input tensor, and 'off_value' is the value used for filling locations other than those specified in 'indices' input tensor. "

[[outputs]]
name = "output"
type = "T3"
desc = "Tensor of rank one greater than input tensor 'indices', i.e. rank(output) = rank(indices) + 1. The data type for the elements of the output tensor is the same as the type of input 'values' is used."

[type_constraints.T1]
types = "all_numeric_types"
desc = "Constrain input to only numeric types."

[type_constraints.T2]
types = "all_numeric_types"
desc = "Constrain input to only numeric types."

[type_constraints.T3]
types = "all_tensor_types"
desc = "Constrain to any tensor type."
```

Produces a one-hot tensor based on inputs.
The locations represented by the index values in the 'indices' input tensor will have 'on_value'
and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
are specified as part of required input argument 'values', which is a two-element tensor of format
[off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
input tensor. The additional dimension is for one-hot representation. The additional dimension will
be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional
dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
the range [0, depth) will result in one-hot representation with all 'off_value' values in the
output tensor.
