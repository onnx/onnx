```toml
name = "ReduceMean"
domain = "ai.onnx"

[attributes.keepdims]
type = "INT"
desc = "Keep the reduced dimension or not, default 1 means keep reduced dimension."
default = 1

[attributes.noop_with_empty_axes]
type = "INT"
desc = "Defines behavior when axes is not provided or is empty. If false (default), reduction happens over all axes (similar to the case when `axis=None` in numpy). If true, reduction happens over an empty set of axes (similar to the case when `axis=()` in numpy). Note that reduction over an empty set of axes means that the reduction step behaves like a no-op (identity function), but composite-reduction operators will still perform the non-reduction steps as needed. Thus, ReduceLogSum returns the Log of input tensor, and ReduceSumSquare returns the Square of the input tensor, in this case."
default = 0

[[inputs]]
name = "data"
type = "T"
desc = "An input tensor."
differentiation = "differentiable"

[[inputs]]
name = "axes"
type = "tensor(int64)"
desc = "Optional input list of integers, along which to reduce. The default is to reduce over empty axes. When axes is empty (either not provided or explicitly empty), behavior depends on 'noop_with_empty_axes': reduction over all axes if 'noop_with_empty_axes' is false, and reduction over the empty set of axes when 'noop_with_empty_axes' is true. Accepted range is [-r, r-1] where r = rank(data)."
option = "optional"
differentiation = "non_differentiable"
```
