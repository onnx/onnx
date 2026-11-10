```toml
name = "Slice"
domain = "ai.onnx"

[[inputs]]
name = "data"
type = "T"
desc = "Tensor of data to extract slices from."

[[inputs]]
name = "starts"
type = "Tind"
desc = "1-D tensor of starting indices of corresponding axis in `axes`"

[[inputs]]
name = "ends"
type = "Tind"
desc = "1-D tensor of ending indices (exclusive) of corresponding axis in `axes`"

[[inputs]]
name = "axes"
type = "Tind"
desc = "1-D tensor of axes that `starts` and `ends` apply to. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data)."
option = "optional"

[[inputs]]
name = "steps"
type = "Tind"
desc = "1-D tensor of slice step of corresponding axis in `axes`. Negative value means slicing backward. 'steps' cannot be 0. Defaults to 1."
option = "optional"
```

Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://numpy.org/doc/stable/reference/routines.indexing.html
Slices uses `starts`, `ends`, `axes` and `steps` inputs to specify the start and end
dimension and step for each axis in the list of axes, it uses this information to
slice the input `data` tensor. If a negative value is passed for any of the
start or end indices, it represents number of elements before the end of that
dimension. If the value passed to start or end is larger than the `n` (the
number of elements in this dimension), it represents `n`. For slicing to the
end of a dimension with unknown size, it is recommended to pass in `INT_MAX`
when slicing forward and 'INT_MIN' when slicing backward.
If a negative value is passed for step, it represents slicing backward.
However step value cannot be 0.
If `axes` are omitted, they are set to `[0, ..., ndim-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`
Example 1:
  data = [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
  ]
  axes = [0, 1]
  starts = [1, 0]
  ends = [2, 3]
  steps = [1, 2]
  result = [
      [5, 7],
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
