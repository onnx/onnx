```toml
name = "TopK"
domain = "ai.onnx"
shape_inference = "defs::math::utils::TopKShapeInference"

[attributes.axis]
type = "INT"
desc = "Dimension on which to do the sort. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input)."
default = -1

[attributes.largest]
type = "INT"
desc = "Whether to return the top-K largest or smallest elements."
default = 1

[attributes.sorted]
type = "INT"
desc = "Whether to return the elements in sorted order."
default = 1

[[inputs]]
name = "X"
type = "T"
desc = "Tensor of shape [a_0, a_1, ..., a_{n-1}]"
differentiation = "differentiable"

[[inputs]]
name = "K"
type = "tensor(int64)"
desc = "A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve"
differentiation = "non_differentiable"

[[outputs]]
name = "Values"
type = "T"
desc = "Tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] containing top K values from the input tensor"
differentiation = "differentiable"

[[outputs]]
name = "Indices"
type = "I"
desc = "Tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] containing the corresponding input tensor indices for the top K values."
differentiation = "non_differentiable"

[type_constraints.T]
types = "all_numeric_types"
desc = "Constrain input and output types to numeric tensors."

[type_constraints.I]
types = ["tensor(int64)"]
desc = "Constrain index tensor to int64"
```

Retrieve the top-K largest or smallest elements along a specified axis. Given an input tensor of
shape [a_0, a_1, ..., a_{n-1}] and integer argument k, return two outputs:

* Value tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}]
  which contains the values of the top k elements along the specified axis
* Index tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] which
  contains the indices of the top k elements (original indices from the input
  tensor).

* If "largest" is 1 (the default value) then the k largest elements are returned.
* If "sorted" is 1 (the default value) then the resulting k elements will be sorted.
* If "sorted" is 0, order of returned 'Values' and 'Indices' are undefined.

Given two equivalent values, this operator uses the indices along the axis as
a tiebreaker. That is, the element with the lower index will appear first.
