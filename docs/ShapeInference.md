<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Shape Inference

ONNX provides an optional implementation of shape inference on ONNX
graphs. This implementation covers each of the core operators, as well
as provides an interface for extensibility. Therefore, you may choose
to invoke the existing shape inference functionality on your graphs,
or to define shape inference implementations to go along with your
custom operators (or both!). Shape inference functions are stored as a
member of the OpSchema objects.

In ONNX 1.10 release, symbol generation and propagation along with shape
data propagation was added to ONNX graph level shape inference.
Detailed proposal is [here](proposals/0005-SymbolicShapeInfProposal.md)

## Background

Please see this [section](IR.md#static-tensor-shapes) of IR.md for a review of static tensor shapes.
In particular, a static tensor shape (represented by a `TensorShapeProto`) is distinct from
a runtime tensor shape. This feature is commonly used when the exact runtime tensor shape is
not known statically (that is, at compile time).

* A `Tensor` with an undefined `shape` field is used to represent a tensor of unknown rank.
* A `Tensor` with a defined `shape` represents a tensor of known rank.
* Each `Dimension` of a `TensorShapeProto` can have a known integer value
(represented by the `dim_value` field) or it can have an unknown value
represented by a symbolic identified (the `dim_param` field) or it
may have neither field defined (in which case it represents an anonymous
unknown value).

## Invoking Shape Inference

Shape inference can be invoked either via C++ or Python. The Python
API is described, with example,
[here](PythonAPIOverview.md#running-shape-inference-on-an-onnx-model).

The C++ API consists of a single function

```cpp
shape_inference::InferShapes(
    ModelProto& m,
    const ISchemaRegistry* schema_registry);
```

The first argument is a `ModelProto` to perform shape inference on,
which is annotated in-place with shape information. The second
argument is optional.

## Limitations

Shape inference is not guaranteed to be complete. In particular, some
dynamic behaviors block the flow of shape inference, for example a
Reshape to a dynamically-provide shape. Also, all operators are not
required to have a shape inference implementation.

### Partial data propagation

With `data_prop=True` in Python or
`ShapeInferenceOptions::enable_data_propagation` in C++, partial data propagation
represents INT32 and INT64 tensor elements as known integers, symbolic values,
or unknown values.
These entries describe tensor values, not tensor dimensions; a symbolic
entry need not be nonnegative.
It does not evaluate floating-point values. An INT32 result containing a value
outside the range [-2147483648, 2147483647] or a symbolic value is not propagated.
`Size` propagation uses the input shape regardless of the input element type.
A propagated record is exposed as a concrete tensor only if all stored values
are known integers, the tensor has a known shape with nonnegative integer
dimensions, and their product equals the number of stored values.
Subgraph-local records are not visible to enclosing or sibling graphs;
subgraphs can read visible outer-scope records.
Records whose value count differs from a fully known tensor shape are not
propagated.

For a symbolic `Reshape` target entry, inference retains the symbol only when
the value is proven positive, or proven nonnegative with `allowzero=1`.
With `allowzero=0`, an entry equal to the input dimension at the same axis
is also retained: copying that dimension when the entry is zero gives the same
result. Otherwise, the entry does not determine an output dimension, because
zero and -1 have special meanings.

### Arithmetic and bounds

The symbolic arithmetic helpers support addition, subtraction, multiplication,
and integer division (`Div`, rounding toward zero). `FloorDiv` rounds toward
negative infinity and requires a known positive integer divisor; otherwise
its result is unknown. Computing addition, subtraction, multiplication, or
`Div` on two known integers uses checked signed INT64 arithmetic: overflow or
division by zero is an inference error, not a wrapped constant.

The identities `x + 0`, `0 + x`, `x - 0`, `x * 1`, `1 * x`, and `Div(x, 1)`
retain `x`. Multiplication by zero yields zero even when the other operand is
unknown. Other compound expressions require both operands to be known integers
or named symbols; an anonymous unknown operand leaves the result unknown.
These rules describe arithmetic performed by inference, not a requirement that
every operator compute every representable expression.

The shared graph-level registry derives bounds within one invocation.
Symbols from existing tensor-shape annotations have the closed integer bounds
[0, 9223372036854775807], and a known integer has its own value as both bounds.
Arbitrary partial symbolic values have no bounds unless the same symbol occurs
in such an annotation. Generated expression symbols acquire bounds from their
arithmetic, not merely from being used as output dimensions.
Addition adds corresponding endpoints; subtraction
subtracts the right upper bound from the left lower bound, and the right lower
bound from the left upper bound. Multiplication takes the minimum and maximum
of the four endpoint products. Division by a known positive integer applies
the operation's rounding rule to both endpoints.
A derived interval is retained only when all required operand bounds are
available and all endpoint results fit in signed INT64. Otherwise no bound is
established by that operation, even if an expression symbol can be allocated.
Missing proof does not establish positivity, nonnegativity, or absence of overflow.

For example, let `N` be a symbol from an existing tensor-shape annotation,
with the shared registry available:

| Expression or target | Inference consequence |
|---|---|
| `Div(N, 2) + 1` | Its lower bound is one, so it passes the symbolic `Reshape` positivity condition. |
| `N - 1` | Its lower bound is -1; it is not retained merely as a symbolic `Reshape` target dimension. |
| `N` at the same input axis | It passes the `allowzero=0` same-axis condition, including when `N` is zero. |
| INT64 tensor `N * 4` and exact dimension `N * 4` | The bounds cannot exclude overflow, so the two computations have distinct expression identities. |
| `Div(-1, 2)` and `FloorDiv(-1, 2)` | Their known results are zero and -1, respectively. |

### Symbolic identities

Supported symbolic arithmetic operations represent results using `dim_param`
symbols. For positive divisors, floor division used in pooling and convolution
differs from integer `Div`, which truncates toward zero. The implementation
uses distinct expression identities unless the numerator is proven
nonnegative or the divisor is one.
For example, `Concat` on tensors of shapes `(5, 2)` and `(N, 2)` can produce
`(5 + N, 2)`. ONNX IR treats each `dim_param` string as an opaque, globally
scoped symbol, not as an expression evaluated at runtime. Shape inference
does not solve equations between symbols. Before allocating expression names,
inference reserves existing symbols in the model, including subgraphs and local
functions. When a compound symbolic expression can be represented, the same
operation and ordered operands reuse one symbol within an inference invocation,
across shape inference, data propagation, and subgraphs sharing the registry.
Operand identity distinguishes integer constants from symbolic names;
operand order is retained rather than canonicalized using commutativity.
A readable expression
spelling is used only if it is unclaimed; otherwise a fresh name is allocated.
A matching spelling alone does not establish expression identity or equality
with an existing user symbol.
INT64 tensor arithmetic shares an expression identity with exact dimension
arithmetic only when integer bounds prove the operation cannot overflow.
Otherwise, its symbolic result has a distinct identity.

Expression provenance is not serialized. On a subsequent invocation, every
existing `dim_param` is an opaque atom, including names generated by a previous
invocation. Existing output annotations are retained when a newly inferred
symbol differs; equality with that new symbol is not inferred. Inference
contexts without an expression allocator preserve concrete results and
identity operations but leave new compound symbolic results unknown.

Graph-level inference may assign a fresh symbol to an anonymous unknown
dimension. Such a name identifies an unknown value; it does not imply a
recovered arithmetic expression. An unknown rank remains represented by an
absent shape field.
Generated names have no guaranteed prefix or stable spelling across invocations.

### Implementation limits (informative)

Synthetic records of unknown values are limited to 1000 elements. This is not
a limit on all propagated records or on total inference memory use.
Each subgraph inference copies visible propagation records; memory use can
therefore grow with both record size and subgraph nesting depth.
The concrete-data bridge currently leaves empty records unmaterialized.
When constructing a new arithmetic expression, an operand string longer than
1024 characters leaves the result unknown.
Pooling with `ceil_mode=1` leaves a symbolic spatial output dimension
unknown unless effective `SAME_UPPER` or `SAME_LOWER` auto-padding determines it.

These limitations are a property of the current implementation, not
fundamental constraints - if you are in need of something more
advanced, do let us know!

## Type Inference vs. Shape Inference

**Type inference** (determining the element type of outputs) is typically handled
automatically by the schema's type constraints. When a type constraint variable
(e.g., `"T"`) is shared between an input and an output in the schema definition,
the framework propagates the element type from the input to the output without
any explicit inference code.

However, many existing ops still explicitly call `propagateElemTypeFromInputToOutput`
as a best practice for robustness. This is harmless when type constraints already
cover the case, and ensures correct behavior regardless of how shape inference
is invoked.

Explicit type inference logic in `TypeAndShapeInferenceFunction` is only needed when:
- The output type is determined by an **attribute** rather than an input type
  (e.g., `Cast`, where the `to` attribute specifies the output element type)
- The output type differs from all input types in a way that cannot be expressed
  via shared type constraint variables
- The operator uses **heterogeneous** variadic inputs/outputs (see below)

### Homogeneous vs. Heterogeneous variadic inputs/outputs

The homogeneous/heterogeneous flag applies only to variadic (repeated) inputs or
outputs in the schema definition:

- **Homogeneous** (the default): All repeated arguments must have the same type.
  The type constraint variable constrains them to be identical, and the framework
  enforces and propagates this automatically.
- **Heterogeneous**: Each repeated argument may have a distinct type. The type
  constraint variable only describes the set of *allowed* types — it does not
  constrain different arguments to have the same type. This is used by operators
  like `Loop` and `Scan`, whose carried state variables can have mixed types.

When using heterogeneous variadic arguments, the operator's
`TypeAndShapeInferenceFunction` must explicitly propagate types for each
individual argument, since the framework cannot do it automatically.

**Shape inference**, on the other hand, almost always requires explicit logic,
since output shapes typically depend on input shapes, attributes, or both.

## Implementing Shape Inference For Operators

You can add a shape inference function to your operator's Schema with

```cpp
OpSchema& Opschema::TypeAndShapeInferenceFunction(InferenceFunction inferenceFunction);
```

`InferenceFunction` is defined in
[shape_inference.h](/onnx/defs/shape_inference.h), along with the core
interface struct `InferenceContext` and an assortment of helper
methods. `InferenceContext` is the core struct which is provided to
your inference function. It allows accessing information about the
operator's inputs, and also allows writing out inferred information.

To see numerous examples, search for occurrences of
`TypeAndShapeInferenceFunction` in the codebase. One that is
relatively involved is the implementation for `Concat`, in
onnx/defs/tensor/defs.cc.

Please note the following points when implementing the shape-inference method for
operators to avoid common errors:

* Before accessing the `shape` of any input, the code must check that
the shape is available. If unavailable, it should be treated as a dynamic
tensor whose rank is unknown and handled appropriately. Usually, the
shape-inference logic is guarded by a call to `hasInputShape` or
`hasNInputShapes`.

* Before accessing the `dim_value` or `dim_param` of any dimension, the
code must check if these fields have a value. In particular, the code must
handle the possibility that the dimension may not have a statically
known value.

There are several utility functions in [shape_inference.h](/onnx/defs/shape_inference.h)
to handle various common situations.

* Use `checkInputRank` for inputs that must have a fixed rank. (See the
inference for `RoiAlign` as an example.)

* `unifyInputDim` and `unifyDim` and `updateOutputShape` can be used
when multiple input dims are expected to be the same, and when input
dimensions are propagated to specific output dimensions. (See the inference
for `RoiAlign` for an example.)

* `unifyInputShape` and `unifyInputShapePrefix` are higher-level utilities
built on `unifyInputDim`. They unify all dimensions (or a prefix of dimensions)
of an input in one call, enabling a more declarative style. `unifyInputDim`
remains useful for more complex scenarios where individual dimensions are
accessed selectively.

* Overloaded operators `*` and `/` can be used on symbolic dimensions when output
dimensions are computed from input dimensions using arithmetic. (See the inference
for `SpaceToDepth` for an example.)

These utilities handle missing shapes and dimensions safely.

_Example_: Consider a simple matrix-multiplication op that expects inputs of shape
`[M,K]` and `[K,N]` and returns an output of shape `[M,N]`. This can be coded
up as below:

```cpp
   // Check that input 0 has rank 2 (if its rank is known).
   checkInputRank(ctx, 0, 2);
   // Check that input 1 has rank 2 (if its rank is known).
   checkInputRank(ctx, 1, 2);
   Dim M, K, N;
   // Check various dimensions, handling missing dimensions/shapes safely.
   unifyInputDim(ctx, 0, 0, M);
   unifyInputDim(ctx, 0, 1, K);
   unifyInputDim(ctx, 1, 0, K);
   unifyInputDim(ctx, 1, 1, N);
   updateOutputShape(ctx, 0, {M, N});
```

The same example can be written more concisely using `unifyInputShape`, which
checks rank and unifies all dimensions in one call:

```cpp
   Dim M, K, N;
   unifyInputShape(ctx, 0, {M, K});
   unifyInputShape(ctx, 1, {K, N});
   updateOutputShape(ctx, 0, {M, N});
```
