```toml
name = "Slice"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::SliceShapeInference_opset13"
data_propagation = "defs::tensor::utils::SliceDataPropagation_opset13"

[[inputs]]
name = "data"
type = "T"
desc = "Tensor of data to extract slices from."
differentiation = "differentiable"

[[inputs]]
name = "starts"
type = "Tind"
desc = "1-D tensor of starting indices of corresponding axis in `axes`"
differentiation = "non_differentiable"

[[inputs]]
name = "ends"
type = "Tind"
desc = "1-D tensor of ending indices (exclusive) of corresponding axis in `axes`"
differentiation = "non_differentiable"

[[inputs]]
name = "axes"
type = "Tind"
desc = "1-D tensor of axes that `starts` and `ends` apply to. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data). Behavior is undefined if an axis is repeated."
option = "optional"
differentiation = "non_differentiable"

[[inputs]]
name = "steps"
type = "Tind"
desc = "1-D tensor of slice step of corresponding axis in `axes`. Negative value means slicing backward. 'steps' cannot be 0. Defaults to 1s."
option = "optional"
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Sliced data tensor."
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to all tensor types."

[type_constraints.Tind]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain indices to integer types"
```

Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://numpy.org/doc/stable/user/basics.indexing.html?highlight=slice#slicing-and-striding

Slice uses the `starts`, `ends`, `axes` and `steps` inputs to select a sub-tensor
of its input `data` tensor.

An effective `starts[i]`, `ends[i]`, and `steps[i]` must be computed for each `i`
in `[0, ... r-1]` where `r = rank(input)` as follows:

If `axes` are omitted, they are set to `[0, ..., r-1]`.
If `steps` are omitted, they are set to `[1, ..., 1]` of length `len(starts)`

The effective values are initialized as `starts[i] = 0`, `ends[i] = dims[i]` where
`dims` are the dimensions of `input` and `steps[i] = 1`.

All negative elements of `axes` are made non-negative by adding `r` to them, where
`r =rank(input)`.

All negative values in `starts[i]` and `ends[i]` have `dims[axes[i]]` added to them,
where `dims` are the dimensions of `input`. Then `starts[axes[i]]` is clamped to
range `[0, dims[axes[i]]]` for positive stepping, or to range `[0, dims[axes[i]]-1]`
for negative stepping.

The clamping for the adjusted `ends[i]` depends on the sign of `steps[i]` and must
accommodate copying 0 through `dims[axes[i]]` elements, so for positive stepping
`ends[axes[i]]` is clamped to `[0, dims[axes[i]]]`, while for negative stepping it
is clamped to `[-1, dims[axes[i]]-1]`.

Finally, `steps[axes[i]] = steps[i]`.

For slicing to the end of a dimension with unknown size, it is recommended to pass
in `INT_MAX` when slicing forward and 'INT_MIN' when slicing backward.

Example 1:

```
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
```

Example 2:

```
data = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]
starts = [0, 1]
ends = [-1, 1000]
result = [
    [2, 3, 4],
]
```
