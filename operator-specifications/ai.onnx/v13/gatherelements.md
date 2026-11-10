```toml
name = "GatherElements"
domain = "ai.onnx"

[[inputs]]
name = "data"
type = "T"
desc = "Tensor of rank r >= 1."
differentiation = "differentiable"

[[inputs]]
name = "indices"
type = "Tind"
desc = "Tensor of int32/int64 indices, with the same rank r as the input. All index values are expected to be within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds."
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Tensor of the same shape as indices."
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to any tensor type."

[type_constraints.Tind]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain indices to integer types"
```

GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
and an optional attribute `axis` that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). It is an indexing operation
that produces its output by indexing into the input data tensor at index
positions determined by elements of the `indices` tensor.
Its output shape is the same as the shape of `indices` and consists of one value
(gathered from the `data`) for each element in `indices`.

For instance, in the 3-D case (r = 3), the output produced is determined
by the following equations:
```
out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,
```

This operator is also the inverse of ScatterElements. It is similar to Torch's gather operation.

Example 1:
```
data = [
    [1, 2],
    [3, 4],
]
indices = [
    [0, 0],
    [1, 0],
]
axis = 1
output = [
    [1, 1],
    [4, 3],
]
```
Example 2:
```
data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
indices = [
    [1, 2, 0],
    [2, 0, 0],
]
axis = 0
output = [
    [4, 8, 3],
    [7, 2, 3],
]
```
