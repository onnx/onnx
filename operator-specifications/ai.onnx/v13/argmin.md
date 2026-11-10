```toml
name = "ArgMin"
domain = "ai.onnx"

[[inputs]]
name = "data"
type = "T"
desc = "An input tensor."
differentiation = "non_differentiable"

[[outputs]]
name = "reduced"
type = "tensor(int64)"
desc = "Reduced output tensor with integer data type."
differentiation = "non_differentiable"

[type_constraints.T]
types = "all_numeric_types_ir4"
desc = "Constrain input and output types to all numeric tensors."
```

Computes the indices of the min elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the min
is selected if the min appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.
