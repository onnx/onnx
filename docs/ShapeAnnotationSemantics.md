<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Runtime Semantics of Shape Annotations

This document defines what a static shape annotation (see
[Static tensor shapes](IR.md#static-tensor-shapes)) *means* at runtime: the
condition under which the annotation attached to a graph input, graph
output, or intermediate value (`value_info`) is satisfied by a particular
execution of a model.

Stating this precisely serves two purposes:

* it gives implementers of runtimes and backends an unambiguous contract to
  either enforce or safely assume;
* it gives a ground truth against which the *static* analyses (the checker
  and shape inference) can be judged: an inference result is sound exactly
  when it never asserts something that this runtime semantics would not
  guarantee.

This document covers only today's representation, in which a
`TensorShapeProto.Dimension` is either a constant (`dim_value`) or a bare
symbolic name (`dim_param`), and in which — as [IR.md](IR.md#static-tensor-shapes)
notes — dimension variables are not scoped: a `dim_param` with a given name
denotes the same value everywhere it occurs in a model, including inside
nested subgraphs. Extending this semantics to a model with locally scoped
dimension variables (for example, a name local to one iteration of a `Loop`
body, or local to one element of a `Sequence`) is future work and is not
addressed here.

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

## Relationship to static inference and the checker

This runtime semantics is two-valued: for a given run, an annotation either
holds or it is violated. Static shape inference and the checker, by
contrast, must decide — without running the model — whether an annotation
is guaranteed to hold for *every* run consistent with the declared input
annotations. That is undecidable in general, so any static analysis is
necessarily an approximation, and it can either:

* prove an annotation holds for every run (a sound positive result);
* prove an annotation can never hold for any run, independent of input
  values (a genuine model inconsistency, distinct from a runtime failure);
  or
* fail to decide either way.

An inference or checking implementation should not conflate the third case
with either of the first two. In particular, failing to *prove* that two
dimensions are equal is not the same as proving them *unequal*, and a
sound implementation must not report an error in that case. Conversely, an
implementation (or a backend consuming its results, e.g. to decide that two
tensors may safely share a memory buffer) should not treat an *unproven*
equality as an established fact.

In practice these two conservative choices are in tension, and different
consumers of shape information want different defaults: the checker
generally prefers to avoid reporting a mismatch it cannot actually prove
(to avoid rejecting valid models), while a backend that wants to optimize
based on inferred shapes needs the opposite bias, treating an unproven
equality as not yet established. Existing ONNX shape inference mostly
leans toward the latter (conservative) bias, but does not fully achieve it
in the presence of conditional branches — particularly inside loops, where
a branch may not execute on every iteration — which is a known source of
inference results that do not, in fact, hold for every run. This document
does not resolve that gap; it defines the semantics against which such gaps
can be identified and measured.

## What this does not require of a runtime

An actual runtime or backend is **not** required to perform the checks
described above. Most production runtimes skip them for performance, and
instead treat declared annotations (and any inference or checker results
derived from them) as trusted preconditions — for example, to decide that
two tensors with the same declared shape and disjoint lifetimes can share a
memory buffer. If a model or an input violates an annotation that the
runtime trusted, and the runtime performed such an optimization based on
that trust, undefined behavior (such as an invalid memory access) can
result.

It remains the runtime's responsibility to guarantee **safe** execution.
This does not require performing the checks in this document; a runtime may
instead perform the checks only in a "safe" or debug mode, establish
sufficient safety in some other way (for example, bounds-checked buffer
accesses regardless of the declared shape), or accept the risk for a class
of trusted, pre-validated models. This document defines what the checks
would establish if performed; it does not mandate that a runtime perform
them.

The ONNX reference implementation, `onnx.reference.ReferenceEvaluator`,
provides an optional, opt-in implementation of these checks: pass
`check_shape_annotations=True` when constructing a `ReferenceEvaluator` (or
when calling `run`) to validate every input and computed value against its
declared shape annotation, raising a descriptive error on the first
violation.
