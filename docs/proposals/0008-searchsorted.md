<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->
- Feature Name: Add SearchSorted operator to `ai.onnx`
- Start Date: 2026-04-02
- RFC PR: [onnx/onnx#8265](https://github.com/onnx/onnx/pull/7646)
- Status: under discussion
- Authors:
  - cbourjau


## Add a `SearchSorted` operator to the ai.onnx domain

This RFC proposes adding the `SearchSorted` operator to the ai.onnx domain.
Given two tensors `x1` and `x2`, `SearchSorted` finds the indices into `x1` such that, if the corresponding elements in `x2` were inserted before the indices, the order of `x1`, when sorted in ascending order, would be preserved.
The proposed semantics are identical to the `searchsorted` function defined in the [array-API](https://data-apis.org/array-api/draft/API_specification/generated/array_api.searchsorted.html#searchsorted).
`SearchSorted` is proposed as a function operator with a `Scan`-based function body as fallback (see [Using `Scan`](#using-scan)).

## Motivation

An operation of this kind is common in tensor libraries (see [Prior Art section](#prior-art)), but cannot be expressed efficiently in the ONNX standard today.
The operation is useful for feature discretization/bucketization in preprocessing steps (e.g., scikit-learn's [KBinsDiscretizer](https://github.com/scikit-learn/scikit-learn/blob/1074736921eecc3ba84743404696bdcaf877c023/sklearn/preprocessing/_discretization.py#L455)).
`SearchSorted` can also be used to implement a Pandas-like [`merge_asof`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge_asof.html#pandas-merge-asof) operation: for each key from the left data frame (`x2`), find the index of the closest preceding key in the (sorted) right data frame (`x1`).

Thus, a specialized and standardized operator enables exporting more models into a portable and clean computational graph and more efficient execution.

## Proposed specification

The following description is based on the array-api documentation and has been adapted to fit the ONNX standard.

```
Finds the indices into `x1` such that, if the corresponding elements in `x2` were inserted before the indices, the order of `x1`, when sorted in ascending order, would be preserved.

Let `v` be an element of `x2` given by `v = x2[j]`.

  - If `v` is less than all elements in `x1`, then `out[j]` **must** be `0`.
  - If `v` is greater than all elements in `x1`, then `out[j]` **must** be `M`, where `M` is the number of elements in `x1`.
  - Otherwise, each returned index `i = out[j]` **must** satisfy an index condition:

    - If `side == 'left'`, then `x1[i-1] < v <= x1[i]`.
    - If `side == 'right'`, then `x1[i-1] <= v < x1[i]`.

**Notes**

For real-valued floating-point tensors, NaNs are ordered after all non-NaN values and are equivalent to each other, and `-0` and `+0` are equivalent (as in NumPy).

### Attributes

**`side`**: string

Either `"left"` (default) or `"right"`. Controls which index is returned if a value lands exactly on an edge.

### Inputs

**`x1`**: T
Input tensor. **Must** be one-dimensional. If `sorter` is not given, **must** be sorted in ascending order; otherwise, `sorter` **must** be a tensor of indices that sort `x1` in ascending order.

**`x2`**: T
Tensor containing search values.

**`sorter`** (optional): Tind
Tensor of indices that sort `x1` in ascending order. The tensor **must** have the same shape as `x1`.

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

The searching logic of this operator **should** be consistent with other sorting operators of an implementation.

## Drawbacks

This constitutes another operator to the standard, which needs to be tested, implemented, and maintained by downstream runtimes.
However, the operator is fairly easy to implement, and there are only a few corner cases to test.

## Rationale and alternatives

A searchsorted operator can be implemented (inefficiently) using today’s standard in the following ways and circumstances:

### `x2` is a constant

If `x2` is statically known, one may use the `ai.onnx.ml.TreeEnsemble` operator with a single Tree as a binary search.
However, this has several drawbacks:

  - This requires the `ai.onnx.ml` domain, which sees less support in runtimes.
  - The output of that operator is a floating-point value rather than the desired int64 indices.

### `x1` and `x2` have very few elements

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

### Using `Scan`

Lastly, one may implement this operator using Scan, which is the proposed function body.
The `Scan` iterates over `x1` and counts, for each element of `x2`, the elements of `x1` ordered before it (`side == "left"`) or not after it (`side == "right"`):

```
x1_sorted = Gather <axis = 0> (x1, sorter)    # `Identity (x1)` if `sorter` is not given
x2_shape = Shape (x2)
count0 = ConstantOfShape <value = int64[1] {0}> (x2_shape)
x2_eq = Equal (x2, x2)    # false iff NaN
x2_nan = Not (x2_eq)
out = Scan <num_scan_inputs = 1, body = step (count_in, x1_i) => (count_out) {
    x1_eq = Equal (x1_i, x1_i)

    # side == "left"
    lt = Less (x1_i, x2)
    nan_gap = And (x2_nan, x1_eq)
    hit = Or (lt, nan_gap)

    # side == "right"
    x1_nan = Not (x1_eq)
    gt = Less (x2, x1_i)
    nan_gap = And (x1_nan, x2_eq)
    not_hit = Or (gt, nan_gap)
    hit = Not (not_hit)

    hit_i64 = Cast <to = 7> (hit)
    count_out = Add (count_in, hit_i64)
}> (count0, x1_sorted)
```

This implementation has two drawbacks compared to a dedicated kernel:

  - A dedicated operator can optimize this operation (and required allocations) much better than the generic Scan operator.
  - The comparison and summation operations have to process all elements.

## Prior art

The `searchsorted` operation is very common in tensor libraries.
As such, prior art is not difficult to find:

  - [array-api](https://data-apis.org/array-api/draft/API_specification/generated/array_api.searchsorted.html#searchsorted)
  - [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html#numpy-searchsorted)
  - [pytorch](https://docs.pytorch.org/docs/stable/generated/torch.searchsorted.html#torch-searchsorted)
  - [tensorflow](https://www.tensorflow.org/api_docs/python/tf/searchsorted)
  - [jax](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.searchsorted.html)
  - [ndonnx](https://ndonnx.readthedocs.io/en/latest/api/ndonnx.html#ndonnx.searchsorted)

The APIs of all the surveyed functions follow the array-api standard or are a subset thereof (tensorflow).
The pytorch operator offers an additional right argument, but its semantics can be replicated by the (preferred and array-api-compliant) side argument.
NumPy explicitly mentions the use of binary search in its documentation.
Jax’s implementation is noteworthy because it uses a Scan-like approach and allows users to tweak the desired algorithm.
However, such an implementation hint seems overly prescriptive for ONNX.
Ndonnx’s implementation is using one of the above-discussed costly workarounds at the time of writing.

## Unresolved questions

### Required sorting semantics and inter-operator compatibility

It would be nice to use "**must** be consistent with other sorting operators of an implementation", but coupling `SearchSorted` to `Unique` would necessarily require explicit semantics for the `Unique` operator, whose floating point semantics are unspecified today.

## Future possibilities

The searchsorted operation appears mature and well-established.
The here-proposed operator precisely follows the array-api operator without additional functionality.
As such, I see no technical reasons that would prevent us from following future (forward-compatible) updates to the array-api.
