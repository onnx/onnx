```toml
name = "Gather"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "Which axis to gather on. Negative value means counting dimensions from the back. Accepted range is [-r, r-1]"
default = 0

[[inputs]]
name = "data"
type = "T"
desc = "Tensor of rank r >= 1."

[[inputs]]
name = "indices"
type = "Tind"
desc = "Tensor of int32/int64 indices, of any rank q. All index values are expected to be within bounds. It is an error if any of the index values are out of bounds."

[[outputs]]
name = "output"
type = "T"
desc = "Tensor of rank q + (r - 1)."

[type_constraints.T]
types = "all_tensor_types"
desc = "Constrain input and output types to any tensor type."

[type_constraints.Tind]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain indices to integer types"
```

Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).
Example 1:
```
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  indices = [
      [0, 1],
      [1, 2],
  ]
  output = [
      [
          [1.0, 1.2],
          [2.3, 3.4],
      ],
      [
          [2.3, 3.4],
          [4.5, 5.7],
      ],
  ]
```
Example 2:
```
  data = [
      [1.0, 1.2, 1.9],
      [2.3, 3.4, 3.9],
      [4.5, 5.7, 5.9],
  ]
  indices = [
      [0, 2],
  ]
  axis = 1,
  output = [
      [[1.0, 1.9]],
      [[2.3, 3.9]],
      [[4.5, 5.9]],
  ]
```
