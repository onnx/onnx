```toml
name = "TopK"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "Dimension on which to do the sort."
default = -1

[attributes.k]
type = "INT"
desc = "Number of top elements to retrieve"
required = true

[[inputs]]
name = "X"
type = "T"
desc = "Tensor of shape [a_0, a_1, ..., a_{n-1}]"

[[outputs]]
name = "Values"
type = "T"
desc = "Tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] containing top K values from the input tensor"

[[outputs]]
name = "Indices"
type = "I"
desc = "Tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] containing the corresponding input tensor indices for the top K values."

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."

[type_constraints.I]
types = ["tensor(int64)"]
desc = "Constrain index tensor to int64"
```

Retrieve the top-K elements along a specified axis. Given an input tensor of
shape [a_0, a_1, ..., a_{n-1}] and integer argument k, return two outputs:
  -Value tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_0, a_1, ..., a_{axis-1}, k, a_{axis+1}, ... a_{n-1}] which
   contains the indices of the top k elements (original indices from the input
   tensor).
Given two equivalent values, this operator uses the indices along the axis  as
 a tiebreaker. That is, the element with the lower index will appear first.
