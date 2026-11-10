```toml
name = "Gather"
domain = "ai.onnx"

[[inputs]]
name = "data"
type = "T"
desc = "Tensor of rank r >= 1."
differentiation = "differentiable"

[[inputs]]
name = "indices"
type = "Tind"
desc = "Tensor of int32/int64 indices, of any rank q. All index values are expected to be within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds."
differentiation = "non_differentiable"

[[outputs]]
name = "output"
type = "T"
desc = "Tensor of rank q + (r - 1)."
differentiation = "differentiable"

[type_constraints.T]
types = "all_tensor_types_ir4"
desc = "Constrain input and output types to any tensor type."

[type_constraints.Tind]
types = ["tensor(int32)", "tensor(int64)"]
desc = "Constrain indices to integer types"
```

Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).

It is an indexing operation that indexes into the input `data` along a single (specified) axis.
Each entry in `indices` produces a `r-1` dimensional slice of the input tensor.
The entire operation produces, conceptually, a `q`-dimensional tensor of `r-1` dimensional slices,
which is arranged into a `q + (r-1)`-dimensional tensor, with the `q` dimensions taking the
place of the original `axis` that is being indexed into.

The following few examples illustrate how `Gather` works for specific shapes of `data`,
`indices`, and given value of `axis`:
| data shape | indices shape | axis | output shape | output equation |
| --- | --- | --- | --- | --- |
| (P, Q) | ( )  (a scalar)   | 0 | (Q)       | output[q] = data[indices, q] |
| (P, Q, R) | ( )  (a scalar)   | 1 | (P, R)       | output[p, r] = data[p, indices, r] |
| (P, Q) | (R, S) | 0 | (R, S, Q) | output[r, s, q] = data[ [indices[r, s], q] |
| (P, Q) | (R, S) | 1 | (P, R, S) | output[p, r, s] = data[ p, indices[r, s]] |

More generally, if `axis = 0`, let `k = indices[i_{0}, ..., i_{q-1}]`
then `output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]`:

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

If `axis = 1`, let `k = indices[i_{0}, ..., i_{q-1}]`
then `output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]`:

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
