```toml
name = "Slice"
domain = "ai.onnx"

[attributes.axes]
type = "INTS"
desc = "Axes that `starts` and `ends` apply to. It's optional. If not present, will be treated as [0, 1, ..., len(`starts`) - 1]."

[attributes.ends]
type = "INTS"
desc = "Ending indices (exclusive) of corresponding axis in axes`"
required = true

[attributes.starts]
type = "INTS"
desc = "Starting indices of corresponding axis in `axes`"
required = true

[[inputs]]
name = "data"
type = "T"
desc = "Tensor of data to extract slices from."

[[outputs]]
name = "output"
type = "T"
desc = "Sliced data tensor."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to all tensor types."
```

Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://numpy.org/doc/stable/reference/routines.indexing.html
Slices uses `axes`, `starts` and `ends` attributes to specify the start and end
dimension for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represent number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  result = [
      [5, 6, 7],
  ]
Example 2:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  starts = [0, 1]
  ends = [-1, 1000]
  result = [
      [2, 3, 4],
  ]
