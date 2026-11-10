```toml
name = "ArgMax"
domain = "ai.onnx"
shape_inference = "defs::reduction::utils::ArgReduceShapeInference_opset12"

[attributes.axis]
type = "INT"
desc = "The axis in which to compute the arg indices. Accepted range is [-r, r-1] where r = rank(data)."
default = 0

[attributes.keepdims]
type = "INT"
desc = "Keep the reduced dimension or not, default 1 means keep reduced dimension."
default = 1

[attributes.select_last_index]
type = "INT"
desc = "Whether to select the last index or the first index if the {name} appears in multiple indices, default is False (first index)."
default = 0
```

Computes the indices of the max elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equal 0, then the resulting tensor has the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the max
is selected if the max appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.
