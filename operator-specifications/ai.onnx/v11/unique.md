```toml
name = "Unique"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::UniqueShapeInference_opset11"

[attributes.axis]
type = "INT"
desc = "(Optional) The dimension to apply unique. If not specified, the unique elements of the flattened input are returned. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input)."

[attributes.sorted]
type = "INT"
desc = "(Optional) Whether to sort the unique elements in ascending order before returning as output. Must be one of 0, or 1 (default)."
default = 1

[[inputs]]
name = "X"
type = "T"
desc = "A N-D input tensor that is to be processed."
differentiation = "non_differentiable"

[[outputs]]
name = "Y"
type = "T"
desc = "A tensor of the same type as 'X' containing all the unique values or subtensors sliced along a provided 'axis' in 'X', either sorted or maintained in the same order they occur in input 'X'"
differentiation = "non_differentiable"

[[outputs]]
name = "indices"
type = "tensor(int64)"
desc = "A 1-D INT64 tensor containing indices of 'Y' elements' first occurrence in 'X'. When 'axis' is provided, it contains indices to subtensors in input 'X' on the 'axis'. When 'axis' is not provided, it contains indices to values in the flattened input tensor. "
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "inverse_indices"
type = "tensor(int64)"
desc = "A 1-D INT64 tensor containing, for elements of 'X', its corresponding indices in 'Y'. When 'axis' is provided, it contains indices to subtensors in output 'Y' on the 'axis'. When 'axis' is not provided, it contains indices to values in output 'Y'. "
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "counts"
type = "tensor(int64)"
desc = "A 1-D INT64 tensor containing the count of each element of 'Y' in input 'X'"
option = "optional"
differentiation = "non_differentiable"

[type_constraints.T]
types = "all_tensor_types"
desc = "Input can be of any tensor type."
```

Find the unique elements of a tensor. When an optional attribute 'axis' is provided, unique subtensors sliced along the 'axis' are returned.
Otherwise the input tensor is flattened and unique values of the flattened tensor are returned.

This operator returns the unique values or sliced unique subtensors of the input tensor and three optional outputs.
The first output tensor 'Y' contains all unique values or subtensors of the input.
The second optional output tensor 'indices' contains indices of 'Y' elements' first occurrence in 'X'.
The third optional output tensor 'inverse_indices' contains, for elements of 'X', its corresponding indices in 'Y'.
The fourth optional output tensor 'counts' contains the count of each element of 'Y' in the input.

Outputs are either sorted in ascending order or optionally in the order of the first occurrence of the values in the input.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html

Example 1:
```
input_X = [2, 1, 1, 3, 4, 3]
attribute_sorted = 0
attribute_axis = None
output_Y = [2, 1, 3, 4]
output_indices = [0, 1, 3, 4]
output_inverse_indices = [0, 1, 1, 2, 3, 2]
output_counts = [1, 2, 2, 1]
```

Example 2:
```
input_X = [[1, 3], [2, 3]]
attribute_sorted = 1
attribute_axis = None
output_Y = [1, 2, 3]
output_indices = [0, 2, 1]
output_inverse_indices = [0, 2, 1, 2]
output_counts = [1, 1, 2]
```

Example 3:
```
input_X = [[1, 0, 0], [1, 0, 0], [2, 3, 4]]
attribute_sorted = 1
attribute_axis = 0
output_Y = [[1, 0, 0], [2, 3, 4]]
output_indices = [0, 2]
output_inverse_indices = [0, 0, 1]
output_counts = [2, 1]
```

Example 4:
```
input_x = [[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
            [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]]
attribute_sorted = 1
attribute_axis = 1
```

intermediate data are presented below for better understanding:
there are 4 subtensors sliced along axis 1 of input_x (shape = (2, 4, 2)):
```
A: [[1, 1], [1, 1]],
   [[0, 1], [0, 1]],
   [[2, 1], [2, 1]],
   [[0, 1], [0, 1]].
```

there are 3 unique subtensors:
```
[[1, 1], [1, 1]],
[[0, 1], [0, 1]],
[[2, 1], [2, 1]].
```

sorted unique subtensors:
```
B: [[0, 1], [0, 1]],
   [[1, 1], [1, 1]],
   [[2, 1], [2, 1]].
```

output_Y is constructed from B:
```
[[[0. 1.], [1. 1.], [2. 1.]],
 [[0. 1.], [1. 1.], [2. 1.]]]
```

output_indices is to map from B to A:
```
[1, 0, 2]
```

output_inverse_indices is to map from A to B:
```
[1, 0, 2, 0]
```

output_counts:
```
[2, 1, 1]
```
