<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Semantics of Shape Annotations

ONNX models may contain (type and) shape annotations attached to graph inputs, graph
outputs, or intermediate values.
* They serve as documentation. Users have a much easier time understanding
models when they have type and shape annotations.
* They help catch various runtime errors through the static checker.
* They enable runtime optimizations. Inference performance for modern large
(DNN) models largely hinges on statically known shapes of tensors, which are
the basis for memory allocation, memory reuse, static memory planning, etc.

This document covers only today's representation, in which a
`TensorShapeProto.Dimension` is either a constant (`dim_value`) or a bare
symbolic name (`dim_param`), and in which — as [IR.md](IR.md#static-tensor-shapes)
notes — dimension variables are not scoped: a `dim_param` with a given name
denotes the same value everywhere it occurs in a model, including inside
nested subgraphs.

Annotations attached to the model's inputs are _preconditions_: it is the
responsibility of the caller to supply inputs that satisfy the given annotations.
The annotations attached to intermediate values and output values are _assertions_
that are expected to hold true in any inference run (assuming that the inputs
satisfy the preconditions).

The primary complication in treating shape annotations as runtime assertions
is in handling the symbolic dimensions.
This document defines what a static shape annotation (see
[Static tensor shapes](IR.md#static-tensor-shapes)) *means* at runtime: the
condition under which the annotation attached to a graph input, graph
output, or intermediate value (`value_info`) is satisfied by a particular
execution of a model.
Stating this precisely serves several purposes:
* it gives a ground truth against which the *static* analyses (the checker
  and shape inference) can be judged: the checker statically determines if
  the shape annotations may fail during execution, while inference infers
  or improves a given annotation without causing any new failure of the
  annotations.
* it serves as the foundation for soundly extending the shape annotation
  mechanism to support locally scoped dimension variables (for example,
  a name local to one iteration of a `Loop` body, or local to one element
  of a `Sequence`), which is future work and is not addressed here.
* it explicates the binding mechanism to identify the values of dimension
  variables during an inference run, which is helpful in implementing
  some memory allocation optimizations.

## Setup

Consider one execution of a model — one *inference run* — for a fixed set
of input values. Define a **binding map** `β`, a partial function from
symbolic dimension names (strings) to non-negative integers. `β` starts
empty and grows monotonically over the course of the run; an existing entry
is never overwritten, only added to.

A **shape annotation** is a `TensorShapeProto` attached to some value (a
graph input, a graph output, or an intermediate value described by
`value_info`) of known rank, giving each axis either a `dim_value` or a
`dim_param`.

## Checking a value against its annotation

Whenever a value `v` (with actual runtime shape `s = (s_0, ..., s_{r-1})`) is
produced during the run — either supplied as a model input or computed as a
node's output — and `v` has a shape annotation `a = (a_0, ..., a_{r-1})` in
the model, the run **checks** `s` against `a`:

1. If `a` does not specify a rank (no shape at all, i.e. the type has no
   `shape` field), the check trivially succeeds.
2. If `a` specifies a rank different from `r`, the check fails.
3. Otherwise, for each axis `i`:
   * if `a_i` is a constant `dim_value = k`, the check requires `s_i == k`;
   * if `a_i` is a symbolic name `dim_param = "N"`:
     * if `β("N")` is already defined, the check requires
       `s_i == β("N")`;
     * otherwise, the check succeeds for this axis and **binds**
       `β("N") := s_i`;
   * if `a_i` has neither `dim_value` nor `dim_param` set, the check
     trivially succeeds for this axis (an anonymous unknown dimension
     asserts nothing).

If every axis's check succeeds, the value satisfies its annotation, and the
(possibly updated) `β` is used to check subsequent values. If any axis's
check fails, the annotation is violated for this run.

This is repeated for every value produced during the run that carries a
shape annotation. Most intermediate values have no annotation
(`value_info` is optional) and are simply not checked; an annotation on an
unchecked value places no runtime obligation.

## Reading: existential quantification per run

Each symbolic name occurring in a model can be read as **existentially
quantified once per inference run**: "there exists a non-negative integer
`N` such that every axis annotated `N` has that value, for this run."
Different runs (different inputs) may pick different witnesses for `N`;
nothing requires the same name to bind to the same value across different
executions of the model. `β` is exactly a witnessing assignment for that
existential, discovered incrementally as the graph executes.

A constant `dim_value = k` is not existentially quantified: it is a plain
assertion that the axis equals `k` on every run.

## Order-independence

A graph is a DAG, and any execution respects a topological order: a node
only runs after all of its inputs are available. The checking procedure
above can be applied at any point after a value is produced, regardless of
which topological order is chosen, because:

* a value's shape does not change once computed;
* whichever occurrence of a shared symbolic name is checked first *binds*
  it, and every later occurrence merely *verifies* it — but if the
  annotations are mutually consistent, this distinction does not affect
  whether the checks succeed or fail overall, only which occurrence
  happened to bind first.

So whether a model's declared annotations hold for a given run is a
well-defined property of the run, independent of execution order.

## What this does not require of a runtime

An actual runtime or backend is **not** required to perform the checks
as described above; it remains the runtime's responsibility to guarantee
**safe** execution. This does not require performing the checks in this
document; a runtime may instead perform the checks only in a "safe" or
debug mode, and establish sufficient safety in some other way (for example,
bounds-checked buffer accesses regardless of the declared shape).

The ONNX reference implementation, `onnx.reference.ReferenceEvaluator`,
provides an optional, opt-in implementation of these checks: pass
`check_shape_annotations=True` when constructing a `ReferenceEvaluator` (or
when calling `run`) to validate every input and computed value against its
declared shape annotation, raising a descriptive error on the first
violation.
