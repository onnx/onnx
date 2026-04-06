<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->
- Feature Name: Add Searchsorted operator to `ai.onnx`
- Start Date: 2026-04-02
- RFC PR: [onnx/onnx#7646](https://github.com/onnx/onnx/pull/7646)
- Authors:
  - cbourjau


## Add searchsorted to the ai.onnx domain

This RFC proposes adding the `Searchsorted` operator to the ai.onnx domain.
Given two tensors `x1` and `x2`, `Searchsorted` finds the indices into `x1` such that, if the corresponding elements in `x2` were inserted before the indices, the order of `x1`, when sorted in ascending order, would be preserved.
This RFC proposes the addition of an operator that follows the semantics of `searchsorted` as defined in the [array-API][https://data-apis.org/array-api/draft/API_specification/generated/array_api.searchsorted.html#searchsorted] standard.

## Motivation

An operation of this kind is common in tensor libraries (see [Prior Art section](#prior-art)), but cannot be expressed efficiently in the ONNX standard today.
Thus, a specialized and standardized operator enables exporting more models into a portable and clean computational graph and more efficient execution.

## Guide-level explanation

The following description is based on the array-api documentation and has been adapted to fit the ONNX standard.

```
Finds the indices into `x1` such that, if the corresponding elements in `x2` were inserted before the indices, the order of `x1`, when sorted in ascending order, would be preserved.

Let `v` be an element of `x2` given by `v = x2[j]`.

  - If `v` is less than all elements in `x1`, then `out[j]` must be `0`.
  - If `v` is greater than all elements in `x1`, then `out[j]` must be `M`, where `M` is the number of elements in `x1`.
  - Otherwise, each returned index `i = out[j]` must satisfy an index condition:

    - If `side == 'left'`, then `x1[i-1] < v <= x1[i]`.
    - If `side == 'right'`, then `x1[i-1] <= v < x1[i]`.

**Notes**

For real-valued floating-point tensors, the sort order of NaNs and signed zeros is unspecified and thus implementation-dependent.
Accordingly, when a real-valued floating-point tensor contains NaNs and signed zeros, what constitutes ascending order may vary among specification-conforming implementations.

While behavior for tensors containing NaNs and signed zeros is implementation-dependent, specification-conforming implementations should, however, ensure consistency with `Unique`  and other sorting operators.


### Attributes

**`side`**: string

Either `"left"` or `"right"`. Controls which index is returned if a value lands exactly on an edge.

### Inputs

**`x1`**: T
Input tensor. Must be a one-dimensional. If `sorter` is not given, must be sorted in ascending order; otherwise, `sorter` must be a tensor of indices that sort `x1` in ascending order.

**`x2`**: T
Tensor containing search values.

**`sorter`** (optional): Tind
Tensor of indices that sort `x1` in ascending order. The tensor must have the same shape as `x1`.

### Outputs

**`out`**: Tind
A tensor of indices with the same shape as `x2`.

### Type Constraints

**`T`**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
**`Tind`**: tensor(int64)
```

## Reference-level explanation

### Downstream implementations

While not mandating the implementation details, it is advised that runtimes implement this operator using a binary search for an `x2` with a large or unknown number of elements.

### Interaction with other operators

The searching logic of this operator should be consistent with other sorting operators of an implementation.
Sorting of floating-point numbers must follow IEEE 754-2019 total-ordering predicate (wikipedia).

## Drawbacks

This constitutes another operator to the standard, which needs to be tested, implemented, and maintained by downstream runtimes. However, the operator is fairly easy to implement, and there are only a few corner cases to test.

## Rationale and alternatives

A searchsorted operator can be implemented (inefficiently) using today’s standard in the following ways and circumstances:

### `x2` is a constant

If `x2` is statically known, one may use the `ai.onnx.ml.TreeEnsemble` operator with a single Tree as a binary search.
However, this has several drawbacks:

  - This requires the `ai.onnx.ml` domain, which sees less support in runtimes.
  - The output of that operator is a floating-point value rather than the desired int64 indices.

### `x1` and `x2` are very small

If both tensors are quite small, one may opt for an implementation akin to the following array-api code:

```
x1 = x1[:, None]
x2 = x2[None, :]

if side == "left":
    indices = (x1 < x2).astype(int64).sum(axis=0)
else:
    indices = (x1 <= x2).astype(int64).sum(axis=0)
```

However, this requires creating two (boolean and int64) matrices of size `len(x1)` x `len(x2)` and performing a subsequent full traversal for the summation.

## Using `Scan`

Lastly, one may implement this operator using Scan.
The loop body may be given as:

```
def body(x1_item):
    if side == "left":
        return (x1 < x2).astype(int64).sum(axis=0)
    else:
        return (x1 <= x2).astype(int64).sum(axis=0)
```

which would be called for each item in `x1` (capturing `x2`).
This implementation has two drawbacks:

  - A dedicated operator can optimize this operation (and required allocations) much better than the generic Scan operator.
  - The comparison and summation operations have to process all elements.

## Prior art

The `searchsorted` operation is a very common in tensor libraries.
As such, prior art is not difficult to find:

  - [array-api](https://data-apis.org/array-api/draft/API_specification/generated/array_api.searchsorted.html#searchsorted)
  - [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html#numpy-searchsorted)
  - [pytorch](https://docs.pytorch.org/docs/stable/generated/torch.searchsorted.html#torch-searchsorted)
  - [tensorflow](https://www.tensorflow.org/api_docs/python/tf/searchsorted))
  - [jax](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.searchsorted.html)
  - [ndonnx](https://ndonnx.readthedocs.io/en/latest/api/ndonnx.html#ndonnx.searchsorted)

The APIs of all the surveyed functions follow the array-api standard or are a subset thereof (tensorflow).
The pytorch operator offers an additional right argument, but its semantics can be replicated by the (preferred and array-api-compliant) side argument.
NumPy explicitly mentions the use of binary search in its documentation.
Jax’s implementation is noteworthy because it uses a Scan-like approach and allows users to tweak the desired algorithm.
However, such an implementation hint seems overly prescriptive for ONNX.
Ndonnx’s implementation is using one of the above-discussed costly workarounds at the time of writing.

## Unresolved questions

  - Floating point sorting behavior (nans and signed-zeros)

## Future possibilities

The searchsorted operation appears mature and well-established.
The here-proposed operator precisely follows the array-api operator without additional functionality.
As such, I see no technical reasons that would prevent us from following future (forward-compatible) updates to the array-api.
