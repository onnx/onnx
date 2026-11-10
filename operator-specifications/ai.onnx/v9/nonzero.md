```toml
name = "NonZero"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::NonZeroShapeInference_opset9"

[[inputs]]
name = "X"
type = "T"
desc = "input"

[[outputs]]
name = "Y"
type = "tensor(int64)"
desc = "output"

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain to all tensor types."
```

Returns the indices of the elements that are non-zero
(in row-major order - by dimension).
NonZero behaves similar to numpy.nonzero:
https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html,
but for scalar input, NonZero produces output shape (0, N) instead of (1, N), which is different from Numpy's behavior.
