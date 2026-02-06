<!--
Copyright (c) ONNX Project Contributors
-->

<!--- SPDX-License-Identifier: Apache-2.0 -->
- Feature Name: Add Searchsorted operator to `ai.onnx`
- Start Date: 2026-04-02
- RFC PR: [onnx/onnx#7646](https://github.com/onnx/onnx/pull/7646)
- Authors:
  - cbourjau


## Add `searchsorted` to the `ai.onnx` domain
[summary]: #summary

This RFC proposes the addition of the `Searchsorted` operator to the `ai.onnx` domain.
Given two tensors `x1` and `x2` `searchsorted` finds the indices into `x1` such that, if the corresponding elements in `x2` were inserted before the indices, the order of `x1`, when sorted in ascending order, would be preserved.
This RFC proposes the addition of an operator that follows the semantics of `searchsorted` as defined in the [array-API][https://data-apis.org/array-api/draft/API_specification/generated/array_api.searchsorted.html#searchsorted] standard.

## Motivation
[motivation]: #motivation

An operation of this kind is common in tensor libraries (see Prior Art section) but cannot be expressed efficiently in the ONNX standard today.
Thus, a specialized and standardized operator allows for a portable and cleaner computational graph and a more efficient execution.


## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

The following description is based on the array-api [documentation](https://data-apis.org/array-api/draft/API_specification/generated/array_api.searchsorted.html#searchsorted) and has been adapted to fit the ONNX standard.

```
Finds the indices into ``x1`` such that, if the corresponding elements in ``x2`` were inserted before the indices, the order of ``x1``, when sorted in ascending order, would be preserved.

**Notes**

For real-valued floating-point tensors, the sort order of NaNs and signed zeros is unspecified and thus implementation-dependent.
Accordingly, when a real-valued floating-point tensor contains NaNs and signed zeros, what constitutes ascending order may vary among specification-conforming tensor libraries.

While behavior for tensors containing NaNs and signed zeros is implementation-dependent, specification-conforming implementations should, however, ensure consistency with the ``Unique`` operator and other sorting-operators.


### Attributes

**`side`**: string

Either "left" or 'right'. Controls which index is returned if a value lands exactly on an edge.

Let ``v`` be an element of ``x2`` given by ``v = x2[j]``, where ``j`` refers to a valid index (see :ref:`indexing`).

  - If ``v`` is less than all elements in ``x1``, then ``out[j]`` must be ``0``.
  - If ``v`` is greater than all elements in ``x1``, then ``out[j]`` must be ``M``, where ``M`` is the number of elements in ``x1``.
  - Otherwise, each returned index ``i = out[j]`` must satisfy an index condition:

    - If ``side == 'left'``, then ``x1[i-1] < v <= x1[i]``.
    - If ``side == 'right'``, then ``x1[i-1] <= v < x1[i]``.

### Inputs

**`x1`**: T
Input tensor. Must be a one-dimensional. If ``sorter`` is not given, must be sorted in ascending order; otherwise, ``sorter`` must be a tensor of indices that sort ``x1`` in ascending order.

**`x2`**: T
Tensor containing search values.

**`sorter`** (optional): Tind
Tensor of indices that sort ``x1`` in ascending order. The tensor must have the same shape as ``x1`.

### Outputs

**`out`**: Tind
A tensor of indices with the same shape as ``x2``.

### Type Constraints

**`T`**: tensor(uint8), tensor(uint16), tensor(uint32), tensor(uint64), tensor(int8), tensor(int16), tensor(int32), tensor(int64), tensor(float16), tensor(float), tensor(double), tensor(bfloat16)
**`Tind`**: tensor(int64)
```


## Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

### Downstream implementations 

While not mandating the implementation details, it is advised that runtimes implement this operator using a binary search for an `x2` with a large or unknown number of elements.

### Interaction with other operators

The searching logic of this operator should be consistent with other searching-operators of an implementation. 
Sorting of floating point number should follow IEEE 754-2019 total-ordering predicate ([wikipedia](https://en.wikipedia.org/wiki/IEEE_754)).
Note that this is a stricter recommendation than that given by the array-api.

## Drawbacks
[drawbacks]: #drawbacks

This constitutes one more operator to the standard which needs to be tested, implemented, and maintained by downstream runtimes. 


## Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

A searchsorted operator can be implemented (inefficiently) using today's standard in the following ways and circumstances:

### `x2` is a constant

If `x2` is statically known one may use the `ai.onnx.ml.TreeEnsemble` operator with a single Tree as a binary search.
However, this as several drawbacks:
  - This requires the `ai.onnx.ml` domain which sees less support in runtimes.
  - The output of that operator is a floating point value rather than the desired int64 indices.
  
### `x1` and `x2` are very small

If both tensors are quite small one may opt for an implementation akin to the following array-api code:
```python
x1 = x1[:, None]
x2 = x2[None, :]

if side == "left":
    indices = (x1 < x2).astype(int64).sum(axis=0)
else:
    indices = (x1 <= x2).astype(int64).sum(axis=0)
```
However, this necessitates the creation of an `len(x1)` x `len(x2)` shaped matrix and a full subsequent traversal for the summation.
In other words, this is a dangerously inefficient implementation.

### Using `Scan`

Lastly, one may implement this operator using `Scan`.
The loop body may be given as:
```python
def body(x1_item):
    if side == "left":
        return (x1 < x2).astype(int64).sum(axis=0)
    else:
        return (x1 <= x2).astype(int64).sum(axis=0)
	
```
which would be called for each item in `x1` (capturing `x2`).
This implementation has two drawbacks:
   - A dedicated operator can optimize this operation (and required allocations) much better than the generic `Scan` operator.
   - The comparison an summation operations have to process all elements.

## Prior art
[prior-art]: #prior-art

The `searchsorted` operator is a fairly common operation in tensor libraries. 
As such, prior art is not difficult to find:

  - [array-api](https://data-apis.org/array-api/draft/API_specification/generated/array_api.searchsorted.html#searchsorted)
  - [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html#numpy-searchsorted)
  - [pytorch](https://docs.pytorch.org/docs/stable/generated/torch.searchsorted.html#torch-searchsorted)
  - [tensorflow](https://www.tensorflow.org/api_docs/python/tf/searchsorted))
  - [jax](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.searchsorted.html)
  - [ndonnx](https://ndonnx.readthedocs.io/en/latest/api/ndonnx.html#ndonnx.searchsorted)

The APIs of all the surveyed functions follow the array-api standard or are a subset thereof (tensorflow).
The pytorch operator offers an additional `right` argument but its semantics can be replicated by the (preferred and array-api-compliant) `side` argument.
NumPy explicitly points out the use of a binary search in the documentation.
Jax's implementation is noteworthy since it uses a `Scan`-like [implementation](https://github.com/jax-ml/jax/blob/b23e3a44dc45f58679db93818c9a98632323e130/jax/_src/numpy/lax_numpy.py#L9470-L9475) and allows the user to tweak the desired algorithm.
However, such an implementation hint seems overly prescriptive for ONNX.
Ndonnx's implementation is [using](https://github.com/Quantco/ndonnx/blob/d6438cac3912c74d130b3c2f38b073bef1c551e4/ndonnx/_typed_array/onnx.py#L834-L859) one of the above-discussed costly workarounds at the time of writing.

## Unresolved questions
[unresolved-questions]: #unresolved-questions

- Floating point sorting behavior (nans and signed-zeros)

## Future possibilities
[future-possibilities]: #future-possibilities

The searchsorted operation appears mature and well established. 
The here-proposed operator precisely follows the array-api operator without additional functionality. 
As such we have no technical reason that would prevent us from following future (forward compatible) updates of the array-api.


