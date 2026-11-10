```toml
name = "Compress"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "(Optional) Axis along which to take slices. If not specified, input is flattened before elements being selected."

[[inputs]]
name = "input"
type = "T"
desc = "Tensor of rank r >= 1."

[[inputs]]
name = "condition"
type = "T1"
desc = "Rank 1 tensor of booleans to indicate which slices or data elements to be selected. Its length can be less than the input length alone the axis or the flattened input size if axis is not specified. In such cases data slices or elements exceeding the condition length are discarded."

[[outputs]]
name = "output"
type = "T"
desc = "Tensor of rank r if axis is specified. Otherwise output is a Tensor of rank 1."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to all tensor types."

[type_constraints.T1]
types = ["tensor(bool)"]
desc = "Constrain to boolean tensors."
```

Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
In case axis is not provided, input is flattened before elements are selected.
Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
