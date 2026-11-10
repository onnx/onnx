```toml
name = "Gather"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::GatherShapeInference_opset11"
data_propagation = "GatherOp13DataPropagator"

[attributes.axis]
type = "INT"
desc = "Which axis to gather on. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data)."
default = 0

[[inputs]]
name = "data"
type = "T"
desc = "Tensor of rank r >= 1."

[[inputs]]
name = "indices"
type = "Tind"
desc = "Tensor of int32/int64 indices, of any rank q. All index values are expected to be within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds."
```

Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).

axis = 0 :

Let
k = indices[i_{0}, ..., i_{q-1}]
Then
output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]

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
axis = 1 :

Let
k = indices[i_{0}, ..., i_{q-1}]
Then
output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]

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
