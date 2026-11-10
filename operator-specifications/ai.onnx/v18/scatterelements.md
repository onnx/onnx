```toml
name = "ScatterElements"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "Which axis to scatter on. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data)."
default = 0

[attributes.reduction]
type = "STRING"
desc = "Type of reduction to apply: none (default), add, mul, max, min. 'none': no reduction applied. 'add':  reduction using the addition operation. 'mul': reduction using the multiplication operation.'max': reduction using the maximum operation.'min': reduction using the minimum operation."
default = "none"
```

ScatterElements takes three inputs `data`, `updates`, and `indices` of the same
rank r >= 1 and an optional attribute axis that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). The output of the operation
is produced by creating a copy of the input `data`, and then updating its value
to values specified by `updates` at specific index positions specified by
`indices`. Its output shape is the same as the shape of `data`.

For each entry in `updates`, the target index in `data` is obtained by combining
the corresponding entry in `indices` with the index of the entry itself: the
index-value for dimension = axis is obtained from the value of the corresponding
entry in `indices` and the index-value for dimension != axis is obtained from the
index of the entry itself.

`reduction` allows specification of an optional reduction operation, which is applied to all values in `updates`
tensor into `output` at the specified `indices`.
In cases where `reduction` is set to "none", indices should not have duplicate entries: that is, if idx1 != idx2,
then indices[idx1] != indices[idx2]. For instance, in a 2-D tensor case, the update
corresponding to the [i][j] entry is performed as below:
```
output[indices[i][j]][j] = updates[i][j] if axis = 0,
output[i][indices[i][j]] = updates[i][j] if axis = 1,
```
When `reduction` is set to some reduction function `f`, the update corresponding to the [i][j] entry is performed as below:
```
output[indices[i][j]][j] = f(output[indices[i][j]][j], updates[i][j]) if axis = 0,
output[i][indices[i][j]] = f(output[i][indices[i][j]], updates[i][j]) if axis = 1,
```
where the `f` is `+`, `*`, `max` or `min` as specified.

This operator is the inverse of GatherElements. It is similar to Torch's Scatter operation.

(Opset 18 change): Adds max/min to the set of allowed reduction ops.

Example 1:
```
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
```
Example 2:
```
data = [[1.0, 2.0, 3.0, 4.0, 5.0]]
indices = [[1, 3]]
updates = [[1.1, 2.1]]
axis = 1
output = [[1.0, 1.1, 3.0, 2.1, 5.0]]
```
