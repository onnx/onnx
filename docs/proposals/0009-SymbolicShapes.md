<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->
- Feature Name: symbolic_shapes
- Start Date: 2026-07-08
- RFC PR: [onnx/onnx#0000](https://github.com/onnx/onnx/pull/0000)
- Status: under discussion
- Authors: (list of github user names)

## Summary
[summary]: #summary

The goal is to implement an algorithm supporting symbolic expressions for dynamic dimensions such as ``2*batch``
or ``past_seq+seq``.

## Motivation

LLMs constantly plays with growing dimensions. Knowing how dimensions evolve helps planning memory allocation
during the computation. It can be planned ahead of time.

It is also mandatory to implement a robust fusion algorithm: shapes must be verified
if a user wants to replace a group of nodes by another one equivalent.

## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

This proposal relies on the following components:

* An independent library manipulating symbolic expressions,
* Backend tests to evaluate the proposed algorithm,
* New shape inference functions implementing the logic for every kernel,
* A mechnism to propagate tiny values as shapes (to handle scenarios such as `Reshape(X, Shape(Y))`),
* A mechanism to store the fact that two expressions are the same otherwise the model computation would fail,
* A mechanism to store the fact that a dimension is necessarily less or equal than another (which happens with Compress operator)
* A logging mechanism to trace the decisions made along the inference.

This proposal is actually based on an existing C++ implementation described on page
[Shape inference](https://xadupre.github.io/docs/onnx-light/design/shape_inference/index.html).

## Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

This proposal is based on an existing C++ implementation
[onnx_light/onnx_optim/shapes](https://github.com/xadupre/onnx-light/tree/main/onnx_light/onnx_optim/shapes).

### Symbolic Expressions

### Backend Tests

### Operator shape inference function

### Propagating Value as Shape

### Constraints

### Logging

## Related work

This section summarizes ongoing work that motivates and informs this proposal.
The current proposal is more complete and makes different decisions based on
some issues faced along the experimental implementation.

### Symbolic dimension arithmetic ([onnx/onnx#7661](https://github.com/onnx/onnx/pull/7661))

Introduces symbolic dimension arithmetic to ONNX shape inference so that
operators can compute output shapes as symbolic expressions when their input
dimensions are symbolic. Key points:

- Adds helper functions in `onnx/defs/shape_inference.h` (`dimToString`,
  `wrapIfCompound`, and overloaded `+`, `-`, `*`, `/` operators for `Dim`
  values). Expressions are encoded as `dim_param` strings and handle both
  concrete and symbolic cases.
- Updates pooling and unpooling shape inference (`onnx/defs/nn/defs.cc`,
  `onnx/defs/nn/old.cc`) to emit symbolic output dimensions, including special
  handling for `SAME` `auto_pad` and strides.
- Improves `Concat` and `Repeat` shape inference (`onnx/defs/tensor/defs.cc`,
  `onnx/defs/tensor/old.cc`) to accumulate axis dimensions and repeat symbolic
  dims using the new helpers.
- Enhances `MathOpDataPropagator` (`onnx/defs/math/defs.cc`) to produce
  symbolic expressions for `Add`, `Sub`, and `Mul` when an operand is symbolic,
  and adds/improves `PartialDataPropagationFunction` implementations for `Div`,
  `Neg`, `Relu`, `Ceil`, `Floor`, and `Range`.
- Improves `Loop` shape inference to propagate subgraph output shapes for loop
  state variables and to set the iteration dimension from the trip-count input
  when available.
- Changes the default symbol prefix in `SymbolTable::createNew` from `unk__` to
  `_d` for more readable symbolic dimension names.

### Symbolic shape inference RFC ([onnx/onnx#7692](https://github.com/onnx/onnx/pull/7692))

Adds the RFC document describing the symbolic dimension arithmetic feature above.
It notes that the original ONNX 1.10 symbolic shape proposal (`0005`) explicitly
left symbolic *expressions* as a non-goal, and documents why that gap is now
being closed (unknown conv/pool output dims, lost `Concat` sums, opaque
`Tile`/`Repeat` shapes, and stalled data propagation through shape subgraphs).
It covers:

- The `dim_param`-string encoding convention, the `_d` symbol prefix, and Python
  examples for `MaxPool` and `Concat` with symbolic inputs.
- The C++ helper API and expression formatting rules, plus the new
  `PartialDataPropagationFunction` additions and the `Loop` scan-output
  iteration-dim fix.
- **Unification of symbolic expressions**: how `unifyDim`/`mergeInDimensionInfo`
  handle symbolic ↔ symbolic comparisons — the target is preserved, the source
  is silently discarded, and there is no algebraic equality check (e.g.
  unifying `"N"` with `"N+1"` succeeds silently, and `onnx.checker` does not
  flag it).
- **Broadcast with symbolic expressions**: the current fallback to a
  fully-unknown dimension when two distinct symbolic `dim_param` strings are
  broadcast together, and a discussion of `broadcast(e1, e2)` as a feasible
  symbolic function.
- A **Rationale** on a possible pluggable `DimensionAlgebra` abstraction
  (virtual methods for arithmetic, symbolic functions, and unification/equality
  checking, with a string-based default), deferred to a follow-on extension.
- Drawbacks and prior art (string-vs-protobuf trade-off; XLA/PyTorch `SymInt`
  and MLIR precedents) and open questions around grammar standardization and
  expression simplification.

## Drawbacks
[drawbacks]: #drawbacks

## Unresolved questions
[unresolved-questions]: #unresolved-questions

Should this be part of the current shape inference algorithm or implemented in a separate
place? Having a new shape inference algorithm makes it easier to smoothly move. We could keep
both algorithms and let user switch to the new one after fixing potential issues.

## Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

The current shape inference algorithm fails most of the LLMs and therefore is not useful.
The fact that there exists many existing shape inference algorithms prooves that the current
one is not efficient enough and it is needed. It must be implemented in C++
to be integrated in runtimes.

## Prior art
[prior-art]: #prior-art

The following page compares existing dynamic shapes inference algorithms:
[shape inference coverage](https://xadupre.github.io/dashboard/onnx-light/shape-inference-coverage.html).
The most advanced relies on `sympy`.
