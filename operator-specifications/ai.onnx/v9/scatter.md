```toml
name = "Scatter"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::ScatterShapeInference_opset9"

[attributes.axis]
type = "INT"
desc = "Which axis to scatter on. Negative value means counting dimensions from the back. Accepted range is [-r, r-1]"
default = 0

[[inputs]]
name = "data"
type = "T"
desc = "Tensor of rank r >= 1."

[[inputs]]
name = "indices"
type = "Tind"
desc = "Tensor of int32/int64 indices, of r >= 1 (same rank as input)."

[[inputs]]
name = "updates"
type = "T"
desc = "Tensor of rank r >=1 (same rank and shape as indices)"

[[outputs]]
name = "output"
type = "T"
desc = "Tensor of rank r >= 1 (same rank as input)."

[type_constraints.T]
types = "all_tensor_types"
desc = "Input and output types can be of any tensor type."

[type_constraints.Tind]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain indices to integer types"
```

Given `data`, `updates` and `indices` input tensors of rank r >= 1, write the values provided by `updates`
into the first input, `data`, along `axis` dimension of `data` (by default outer-most one as axis=0) at corresponding `indices`.
For each entry in `updates`, the target index in `data` is specified by corresponding entry in `indices`
for dimension = axis, and index in source for dimension != axis. For instance, in a 2-D tensor case,
data[indices[i][j]][j] = updates[i][j] if axis = 0, or data[i][indices[i][j]] = updates[i][j] if axis = 1,
where i and j are loop counters from 0 up to the respective size in `updates` - 1.
Example 1:
  data = [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
  ]
  indices = [
      [1, 0, 2],
      [0, 2, 1],
  ]
  updates = [
      [1.0, 1.1, 1.2],
      [2.0, 2.1, 2.2],
  ]
  output = [
      [2.0, 1.1, 0.0]
      [1.0, 0.0, 2.2]
      [0.0, 2.1, 1.2]
  ]
Example 2:
  data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
  indices = [[1, 3]]
  updates = [[1.1, 2.1]]
  axis = 1
  output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
