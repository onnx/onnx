<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->
- Feature Name: symbolic_shapes
- Start Date: 2026-07-08
- RFC PR: [onnx/onnx#8175](https://github.com/onnx/onnx/pull/8175)
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

This proposal is based on an existing C++ implementation under
[onnx_light/onnx_optim](https://github.com/xadupre/onnx-light/tree/main/onnx_light/onnx_optim).
Shape inference computes, for every value in a graph, its element type and a
(possibly symbolic) shape without running the model. The engine keeps its
working state in a `ShapesContext`
([shapes/shapes_context.h](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/shapes/shapes_context.h)):
a `name → OptimTensor` map, where each `OptimTensor`
([optim_tensor.h](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/optim_tensor.h))
holds an element type and a shape made of `OptimDim` entries, each either a
concrete integer or a symbolic expression string.

The top-level driver `ShapesContext::ComputeShapeModel`
([shapes/shape_inference.cc](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/shapes/shape_inference.cc),
Python entry point
[shape_inference.py](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/shape_inference.py))
registers opsets and local functions, collects the shapes declared on the graph
outputs and `value_info` as authoritative *anchors*, seeds the context from
initializers and inputs, walks the nodes in topological order, reconciles the
inferred shapes with the anchors, propagates the resulting constraints, and
(optionally) writes the descriptors back as `value_info`.

### Symbolic Expressions

Dynamic dimensions are represented as symbolic expression strings such as
`"2*batch"` or `"cache_length + seq_length"`. Rather than manipulating these
strings with regex substitution or `eval`, the engine parses them into an AST
and applies systematic simplification, evaluation, renaming and arithmetic. The
library lives in
[expressions.h](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/expressions.h)
/
[expressions.cc](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/expressions.cc)
with a documented Python wrapper in
[expressions.py](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/expressions.py).

- **Grammar**: a subset of Python arithmetic — `+`, `-`, `*`, `//` (floor
  division), `%`, plus `^` and `&` re-interpreted as `max` and `min`, and the
  function calls `CeilToInt(n, div)` and `Max(a, b)`. A dedicated `/:` operator
  encodes *exact* integer division (the caller asserts no remainder), which lets
  the simplifier cancel factors across a division where `//` cannot. Its primary
  use is `Reshape`, where the inferred `-1` dimension is always an exact
  quotient.
- **Simplification** applies a fixed pipeline of pure tree-to-tree transformers
  (constant folding, mul/div cancellation, floor-division distribution,
  commutative reordering, ring collapsing) twice, then collects a linear
  combination `{symbol → coefficient}` so that `2*batch//batch → 2` and
  `a + b - a → b`. The simplifier is conservative: it never rewrites unless the
  result holds for *every* integer value of the symbols (hence `(2*H)//2 → H`
  but `2*(H//2)` is left unchanged).
- **Dimension arithmetic** helpers (`dim_add`, `dim_sub`, `dim_mul`, `dim_div`,
  `dim_exact_div`, `dim_mod`, `dim_max`, `dim_min`) compute exactly when both
  operands are integers and otherwise build and simplify an expression string,
  so symbolic arithmetic never accumulates unnormalised intermediates.
- **Renaming** (`rename_expression`, `rename_dynamic_expression`,
  `rename_dynamic_dimensions`) rewrites internal symbols to the user-visible
  names, and is what the constraint pass relies on.

### Backend Tests

Correctness is measured against the ONNX backend test suite. Every backend test
case tagged `"inference"` is run through the pipeline and the *computed* shapes
are compared value-by-value against the *expected* intermediate and output
shapes recorded by the test author. For each case the model's `graph.value_info`
is stripped (so inference cannot simply reuse the recorded shapes),
[infer_shapes_model](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/shape_inference.py)
is run on the stripped clone, and a side-by-side table reports the match status
of every input, intermediate and output. When inference raises, the error is
shown instead. This coverage report is the primary regression signal for the
whole engine and is used to track how many operators produce shapes that exactly
match the expectations.

### Operator shape inference function

Each node is dispatched to a per-operator shape function. `ComputeShapeNode`
first expands **model-local function** calls (by recursively inferring the
function body), then honours any registered **custom callback** for a
`(domain, op_type)` pair, and otherwise looks up the built-in **dispatch table**
([shapes/dispatch_table.cc](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/shapes/dispatch_table.cc),
[shapes/dispatch_table.h](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/shapes/dispatch_table.h))
keyed by `domain:op_type`. The individual operator implementations are grouped by
domain under
[shapes/](https://github.com/xadupre/onnx-light/tree/main/onnx_light/onnx_optim/shapes)
(`math/`, `nn/`, `tensor/`, `reduction/`, `logical/`, `sequence/`,
`controlflow/`, `quantization/`, ...). Before each dispatch the engine checks
that all declared inputs are known and the outputs are not yet defined, so
missing inputs or duplicate definitions are reported early.

Symbolic expressions are introduced here: operators such as `Conv`, `MaxPool`,
`Slice`, `Pad`, `Concat` and `Tile` compute their output dimensions through the
dimension-arithmetic helpers, and every new dimension is simplified before it is
stored. Broadcasting of two shapes is handled by
[shapes/shape_broadcast.cc](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/shapes/shape_broadcast.cc).
Besides tensors, the engine tracks sequence- and map-typed values (`OptimSequence`,
[optim_sequence.h](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/optim_sequence.h))
and infers the nested graphs of `If` / `Loop` / `Scan` in child contexts under
[shapes/controlflow](https://github.com/xadupre/onnx-light/tree/main/onnx_light/onnx_optim/shapes/controlflow),
retaining each child context so the descriptors inferred inside a subgraph stay
inspectable.

### Propagating Value as Shape

Many operators take a **shape tensor** whose runtime *values* determine the
output shape (`Reshape`'s `shape`, `Expand`'s `shape`, `Tile`'s `repeats`,
`Resize`'s `sizes`, ...). Static inference cannot read runtime values in general,
so each `OptimTensor` carries an optional secondary `value_as_shape` annotation
that records the symbolic *content* of an integer tensor. For example
`Shape(x)` on a `[N, C, H, W]` input produces a storage shape `[4]` but a
`value_as_shape` of `[N, C, H, W]`.

- The annotation is **seeded** from integer initializer literals and by the
  `Shape` and `Size` operators.
- It is **propagated** through `Concat` / `Split`, element-wise `Add` / `Sub`
  (via `PropagateValueAsShapeArithmetic` in
  [shapes/shape_broadcast.cc](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/shapes/shape_broadcast.cc),
  which applies `dim_add` / `dim_sub` element-wise), `Gather`, `Slice`,
  `Reshape`, `Squeeze` and `Unsqueeze`, keeping every entry canonical through the
  expression simplifier.
- It is **consumed** by shape-input operators (`Reshape`, `Expand`, `Tile`,
  `Resize`, `ConstantOfShape`, `Pad`, `OneHot`, ...), which read the symbolic
  dimension values directly instead of falling back to a generic output shape.

This lets a `Shape → Concat → Add → Sub → Reshape/Expand` round-trip recover an
exact symbolic output shape such as `[N, 1]`.

### Constraints

Per-operator inference names the symbols it produces locally (e.g. `NonZero`
emits a fresh `NonZero_nz_nnz`), while the model author annotates graph outputs
with their own names (`Z: [2*dnz]`). The constraint store on `ShapesContext`
([shapes/shapes_context.h](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/shapes/shapes_context.h))
bridges the two naming schemes so the inferred `value_info` matches the declared
shapes.

- **Equality constraints** (`a == b`, `AddConstraint`) are recorded when merging
  an inferred shape against an anchor reveals two symbolic expressions must be
  equal. `AddSymbolicConstraintWithLeafDerivation`
  ([shapes/shape_inference.cc](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/shapes/shape_inference.cc))
  also derives leaf equalities from compound anchors — e.g. `2*dnz == 2*nnz`
  yields `dnz == nnz` by asking the simplifier for the algebraic difference of
  the two sides.
- **Upper-bound constraints** (`lhs <= rhs`, `AddLessEqualConstraint`) record
  data-dependent but provably bounded dimensions, e.g. `NonZero`'s `nnz <=
  prod(shape(X))`, `Compress`'s output `count`, or an `If` merge bounded by the
  `max` of the two branches.
- After node inference, `PropagateAnchorConstraintsIntoContext` turns the
  equality constraints into a renaming: it collects the user-declared *preferred*
  names, builds equivalence classes, calls `rename_dynamic_dimensions`, and
  rewrites both `shape` and `value_as_shape` of every tensor to a fixed point —
  so an internally named sibling of `N` becomes `N` while the user's `N` is never
  renamed away.

### Logging

`ShapesContext` carries an **opt-in event log** (disabled by default, so it adds
no overhead) that records what inference did step by step: each descriptor
insertion (`kAdd`) or overwrite (`kReplace`), each node dispatch
(`kComputeNode`), and each recorded constraint (`kConstraint` /
`kConstraintMax`). Every `ShapeEvent`
([shapes/shapes_context.h](https://github.com/xadupre/onnx-light/blob/main/onnx_light/onnx_optim/shapes/shapes_context.h))
carries a `node_index` (with `-1` for graph inputs and `-2` for initializers)
plus a `subgraph_node_index` / `subgraph_attr_name` so an event can be traced
back to the exact node — including inside `If` / `Loop` / `Scan` subgraphs. It is
enabled in Python with `ctx.events_enabled = True` and read through
`ctx.events()`; replaying the log is the easiest way to find which node produced
a given dimension or where an inference error originates.

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
