```toml
name = "GatherND"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::GatherNDShapeInference"

[[inputs]]
name = "data"
type = "T"
desc = "Tensor of rank r >= 1."
differentiation = "differentiable"

[[inputs]]
name = "indices"
type = "Tind"
desc = "Tensor of rank q >= 1. Each index must be in [-s, s-1], where s is the size of the corresponding data axis after the batch dimensions. An out-of-bounds index is an error."
differentiation = "non_differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to any tensor type."

[type_constraints.Tind]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain indices to int32 or int64 tensors."
```

Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.

Let `k = indices_shape[-1]`. Each length-`k` tuple in `indices` selects an element or slice of `data`.

The leading `b` dimensions of `data` and `indices` are batch dimensions. Each tuple indexes
the next `k` dimensions of `data`, starting at axis `b` (axes are zero-based).

The inputs must satisfy the following constraints:

1) `r >= 1` and `q >= 1`.

2) The first `b` dimensions of the shape of `indices` tensor and `data` tensor must be equal.

3) `batch_dims` must satisfy `0 <= b < min(q, r)`.

4) `indices_shape[-1]` must be between 1 and `r-b`, inclusive.

5) For each tuple component `i`, `0 <= i < indices_shape[-1]`, the index must satisfy
   `-data_shape[b+i] <= indices[...,i] < data_shape[b+i]`.
   It is an error if any of the index values are out of bounds.

The output shape is `indices_shape[:-1] + data_shape[b+k:]`, where `+` denotes shape concatenation.
For each coordinate tuple `p` in `indices_shape[:-1]`, the selected value or slice is
`output[p] = data[p[:b] + tuple(indices[p])]`, using tuple concatenation for the data coordinate.
Unspecified trailing data axes select the entire slice. When `k = r-b`, each selected value is a scalar.
Negative indices count backwards from the end of the corresponding data dimension.
Within the same batch, repeated index tuples select the same value or slice at multiple output positions.

**Example 1**

```
batch_dims = 0
data    = [[0,1],[2,3]]   # data_shape    = [2, 2]
indices = [[0,0],[1,1]]   # indices_shape = [2, 2]
output  = [0,3]           # output_shape  = [2]
```

**Example 2**

```
batch_dims = 0
data    = [[0,1],[2,3]]  # data_shape    = [2, 2]
indices = [[1],[0]]      # indices_shape = [2, 1]
output  = [[2,3],[0,1]]  # output_shape  = [2, 2]
```

**Example 3**

```
batch_dims = 0
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]
output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
```

**Example 4**

```
batch_dims = 0
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]
output  = [[[2,3]],[[4,5]]]             # output_shape  = [2, 1, 2]
```

**Example 5**

```
batch_dims = 1
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[1],[0]]                     # indices_shape = [2, 1]
output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
```

Opset 29 adds support for int32 indices; int64 indices remain supported.
