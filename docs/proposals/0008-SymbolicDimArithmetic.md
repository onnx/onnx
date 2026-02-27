<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->
- Feature Name: symbolic-dim-arithmetic
- Start Date: 2026-02-27
- RFC PR: [onnx/onnx#7661](https://github.com/onnx/onnx/pull/7661)
- Status: under discussion
- Authors:
  - justinchuby

# Symbolic Dimension Arithmetic in ONNX Shape Inference

## Summary
[summary]: #summary

This proposal extends ONNX shape inference to support *symbolic dimension arithmetic*: when an operator's output dimension depends on one or more symbolic (unknown, named) input dimensions, shape inference now computes and propagates an expression string (e.g. `"N+2"`, `"(H+pad)/stride"`) rather than leaving the dimension entirely unknown. Expressions are encoded as `dim_param` strings so no protobuf schema changes are required.

## Motivation
[motivation]: #motivation

### Background

ONNX shape inference was extended in ONNX 1.10 ([proposal 0005](0005-SymbolicShapeInfProposal.md)) to add *symbol generation* and *partial data propagation*. That work explicitly listed "adding symbolic expressions to ONNX standard" as a **non-goal** because of implementation complexity. However, production usage revealed that leaving expressions unknown significantly limits the value of shape inference for real-world models.

### Problem statement

When a model uses dynamic (symbolic) dimensions—common in models exported from PyTorch, JAX, or TensorFlow with dynamic batch/sequence sizes—ONNX shape inference frequently loses track of relationships between tensor dimensions:

1. **Derived dimensions become unknown.** A Conv or Pool operator applied to an input of height `H` produces an output of height `(H + pad_top + pad_bottom - kernel_h) / stride_h`. If `H` is symbolic, the current inference leaves the output height entirely unknown, breaking downstream inference.

2. **Concat on symbolic dims loses information.** Concatenating tensors of shapes `[M]` and `[N]` on axis 0 produces shape `[?]` today. The fact that the output length is `M+N` is discarded.

3. **Repeat/Tile cannot propagate.** Repeating a tensor of shape `[N]` three times should give `[3*N]`, but current inference yields `[?]`.

4. **Data propagation stops at dynamic shapes.** Operators like `Div`, `Neg`, `Relu`, `Ceil`, and `Floor` are frequently used in shape-computation subgraphs. Without propagation support for these ops, inferred shapes for downstream nodes stay unknown even when a symbolic expression could be produced.

5. **Loop scan-output iteration dimension is lost.** When a Loop op has a known trip count input `M`, the scan outputs have a leading dimension of `M`, but this is not inferred today.

### Use cases

- **Exporter toolchains (PyTorch, Dynamo, JAX).** Exporters emit models with symbolic batch, sequence, or spatial dimensions. Downstream compilers (MLIR, TensorRT, OnnxRuntime) rely on shape inference to plan memory and validate graphs. Richer inferred shapes reduce the need for runtime shape computation.

- **Memory planning.** A runtime that knows an output shape is `(H+2)/2` rather than `?` can pre-allocate the buffer when `H` is resolved at execution time.

- **Graph optimization.** Pattern matchers that fold `Shape → arithmetic → Reshape` chains work better when symbolic expressions survive inference.

- **Model visualization and debugging.** Seeing `(N*stride - pad)` in a shape tooltip is far more informative than seeing `?`.

## Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

### Symbolic expressions as `dim_param` strings

A dimension in an ONNX `TensorShapeProto` can carry either a concrete integer (`dim_value`) or a named symbol (`dim_param`). This proposal reuses `dim_param` to carry **expression strings** when a dimension cannot be reduced to a concrete value:

```
dim_param = "N + 2"
dim_param = "(H + pad_top + pad_bottom - k_h) / stride_h"
dim_param = "M*3"
```

Tools that only understand simple identifiers treat these as opaque names (the string is still a valid `dim_param`), while tools that understand the new convention can parse and evaluate the expression.

### Default symbol prefix changed to `_d`

New symbols generated during graph-level shape inference now use the prefix `_d` instead of `unk__`. Generated symbols look like `_d0`, `_d1`, etc. This is more compact and less likely to conflict with user-defined names.

### Operators that now produce symbolic expressions

| Operator group | What is inferred |
|---|---|
| Conv, Pool (MaxPool, AveragePool, etc.) | Spatial output dims as `(dim + pads - kernel) / stride` expressions when input dim is symbolic |
| MaxUnpool | Output spatial dims from `input * stride - pads` |
| Concat | Axis dimension as sum of input dims (e.g., `M + N`) |
| Tile / Repeat | Each dimension as `input_dim * repeats` |
| Math ops used in shape subgraphs (Add, Sub, Mul, Div, Neg, Relu, Ceil, Floor) | Output data propagated as symbolic expressions for use by downstream shape inference |
| Range | Data propagated to enable downstream shape uses |
| Loop | First dimension of scan outputs set to trip-count value when it is known |

### Example: MaxPool with symbolic spatial input

```python
import onnx
from onnx import helper, TensorProto

# Input shape: [batch, channels, H, W]  (H, W symbolic)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, ['N', 'C', 'H', 'W'])
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)

node = helper.make_node('MaxPool', ['X'], ['Y'],
                        kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
graph = helper.make_graph([node], 'test', [X], [Y])
model = helper.make_model(graph)

inferred = onnx.shape_inference.infer_shapes(model)
output_shape = inferred.graph.output[0].type.tensor_type.shape
# Before this proposal: shape is [N, C, ?, ?]
# After  this proposal: shape is [N, C, (H + 2 - 3) / 2, (W + 2 - 3) / 2]
#                       which simplifies to [N, C, (H - 1) / 2, (W - 1) / 2]
```

### Example: Concat of symbolic dimensions

```python
# A: [M], B: [N]
# Concat on axis 0 → [M + N]
```

### Example: Data propagation through Ceil/Floor for shape subgraphs

Models that compute output shapes via `Ceil(input_size / stride)` can now have those values propagated, allowing `Reshape` nodes downstream to receive a concrete or symbolic shape rather than `?`.

## Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

### C++ helper API (`onnx/defs/shape_inference.h`)

Three new utility functions are added:

```cpp
// Convert a Dim to a string for use in an expression.
// Returns "" if the dim is unknown (no dim_value or dim_param).
std::string dimToString(const TensorShapeProto::Dimension& dim);

// Returns true if 's' is a simple token (identifier or integer)
// that does not need parentheses as a sub-expression.
bool dimParamIsSimple(const std::string& s);

// Wrap 's' in parentheses if it is compound (contains operators).
std::string wrapIfCompound(const std::string& s);
```

Overloaded arithmetic operators are provided for `TensorShapeProto::Dimension`:

```cpp
Dim operator+(Dim, Dim);      // A + B
Dim operator+(Dim, int64_t);  // A + 2
Dim operator-(Dim, Dim);      // A - B
Dim operator-(Dim, int64_t);  // A - 2
Dim operator*(Dim, Dim);      // A * B
Dim operator*(Dim, int64_t);  // A * 3
Dim operator/(Dim, int64_t);  // A / 2  (integer division; rounds toward zero)
```

Rules:
- If both operands are concrete (`dim_value`), the result is a concrete `dim_value`.
- Identity shortcuts (`*1`, `+0`, `/1`) return the other operand unchanged.
- Zero shortcuts (`*0`) return `dim_value = 0`.
- Otherwise, if both operands have a string representation (either `dim_value` or `dim_param`), the result is a `dim_param` expression string.
- If either operand has no string representation (fully unknown dim), the result is also unknown.

### Expression format

Expressions are infix strings using standard C-like syntax:
- Operator precedence is explicit via parentheses where needed (`wrapIfCompound`).
- Division `/` is integer division (floor towards zero), matching ONNX semantic for shape dimensions.
- No simplification is performed (e.g., `"0 + A"` is left as-is rather than being reduced to `"A"`).

### Symbol generation

`SymbolTable::createNew()` now uses prefix `"_d"` by default, generating names `_d0`, `_d1`, … The old prefix `unk__` is no longer used by the built-in implementation.

### Partial data propagation additions

The following operators now have `PartialDataPropagationFunction` implementations:

| Operator | Behaviour |
|---|---|
| `Div` | Propagates `a / b` as symbolic expression if either operand is symbolic |
| `Neg` | Propagates `-(expr)` |
| `Relu` | Propagates the input unchanged (ReLU is identity for non-negative shape dims) |
| `Ceil` | Propagates `ceil(expr)` as a `dim_param` string |
| `Floor` | Propagates `floor(expr)` as a `dim_param` string |
| `Range` | Propagates the generated range as partial data for downstream use |

### Loop scan-output inference

`LoopInferenceFunction` (and the opset-8, -11, -13 variants) now sets the first dimension of every scan output to the trip-count value when the trip-count input `M` is a known non-negative integer. Previously this dimension was always left unknown.

### Unification of symbolic expressions

Shape inference frequently needs to *unify* two dimensions—asserting that they must be equal. The two relevant functions are `unifyDim` (`onnx/defs/shape_inference.h`) and `mergeInDimensionInfo` (`onnx/defs/shape_inference.h`). Both follow the same priority rules:

| Source dim | Target dim | Result |
|---|---|---|
| concrete value | concrete value | check equality; `fail_shape_inference` if they differ |
| concrete value | symbolic `dim_param` | overwrite target with the concrete value |
| concrete value | unknown (no value/param) | set target to the concrete value |
| symbolic `dim_param` | concrete value | preserve target (concrete beats symbolic) |
| symbolic `dim_param` | symbolic `dim_param` | **preserve target; source is silently discarded** |
| symbolic `dim_param` | unknown | set target to source's `dim_param` string |
| unknown | any | preserve target |

The critical row is **symbolic ↔ symbolic**: the implementation does **not** check whether the two expression strings are algebraically equivalent. It simply keeps whichever string is already on the target side and discards the source. This means:

- If source is `"N + 2"` and target is `"N + 2"` (identical strings), the result is `"N + 2"` — correct, but only by luck of string equality.
- If source is `"N + 2"` and target is `"M + 2"` (different symbols for the same value), the result is `"M + 2"` — no error is raised, but the relationship between the two expressions is lost.
- If source is `"N + 2 - 2"` and target is `"N"` (algebraically equal), the result is `"N"` — again, no error, and the unsimplified form is silently dropped.

This behavior is intentional: performing algebraic equality checking would require a full symbolic solver (out of scope for ONNX shape inference) and would introduce new inference failures in valid models. The trade-off is that inferred shapes may be weaker than theoretically possible in cases where two independently-derived expressions describe the same dimension.

**Implication for new operator implementations.** When writing or updating a shape-inference function that asserts two dimensions must be equal (e.g., batch size must match across two inputs), prefer using `unifyDim(input_dim, output_dim)` where the *target* (`output_dim`) already holds the expression you want to preserve. If neither side has a preferred form, pick the more informative one as the target.

### Interaction with existing shape inference

- All changes are backward compatible: if no symbolic dim is involved, the existing concrete inference path is taken and results are identical to before.
- Expression `dim_param` strings are opaque to tools that do not understand them. They act as a named symbol (dimension identity is by string equality).
- Graph-level shape inference continues to unify dimensions with the same `dim_param` string.

### Corner cases

| Scenario | Result |
|---|---|
| `dim` with neither `dim_value` nor `dim_param` (fully unknown) | Arithmetic on it returns a fully unknown dim |
| Division by zero | Not handled by these helpers; callers must guard against it before calling `/` |
| `SAME` auto-pad | Output dim equals input dim (no expression needed); handled as before |
| Negative padding / stride | Expressions are still generated; values are only validated at runtime |

## Drawbacks
[drawbacks]: #drawbacks

- **Expression strings are opaque to old tools.** A tool that only understands simple identifier `dim_param` values will treat `"N + 2"` as an opaque symbol name. This is safe but may be confusing.
- **No algebraic simplification.** Expressions are not simplified, so chained operations can produce verbose strings like `"((_d0 + 2) - 1) / 2"`. This is a known limitation.
- **Parser burden on consumers.** Tools that want to evaluate expressions must implement a small expression parser. No shared library is provided.
- **Integer semantics only.** Division is integer (truncating) division. Fractional semantics are not supported.

## Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

### Why `dim_param` strings and not a new protobuf field?

Adding a new field to `TensorShapeProto.Dimension` would require a protobuf schema change and a version bump. Encoding expressions in the existing `dim_param` string is 100% backward compatible: old tools see an opaque symbol name and continue to work.

### Why not use a full symbolic algebra library?

A proper CAS (computer algebra system) would support simplification, factoring, and solving equations. This is out of scope for shape inference, which needs to be fast and dependency-free. The expression strings produced here are sufficient for the primary use case (communicating the relationship between symbolic dimensions to compilers and runtimes).

### Alternative: new IR concept "symbolic expression dimension"

An earlier design considered adding a `dim_expr` field to the protobuf. This would be unambiguous and parser-friendly, but requires coordinated changes to the ONNX IR spec, all language bindings, and all downstream tools. The string encoding chosen here defers that specification work to a future version.

### What is the impact of not doing this?

Exporters fall back to generating fully unknown shapes for almost every operator applied to a dynamic input, causing downstream compilers to either fail or produce suboptimal code.

## Prior art
[prior-art]: #prior-art

- **TensorFlow/XLA**: XLA uses a symbolic expression system for shapes (`xla::Shape` with symbolic dimensions represented as `DynamicDimension`). Arithmetic on shapes produces new symbolic values, and a solver can determine dimension equality.
- **PyTorch's `torch.export` / `torch.fx`**: Symbolic shapes are tracked as `SymInt` objects. Arithmetic on `SymInt` values builds an expression tree that is simplified by a constraint solver.
- **MLIR's `ShapedType` with dynamic dimensions**: MLIR uses `-1` to represent dynamic dimensions in IR, but individual dialects (e.g., `tensor.dim`) can carry symbolic expressions in attributes.
- **ONNX-MLIR**: The ONNX-MLIR compiler propagates symbolic shapes through its own pass pipeline, independent of the ONNX runtime shape inference.
- **Existing workaround**: Before this change, users had to run ONNX shape inference with concrete example inputs (via `onnxruntime.InferenceSession`) or use external tools like `onnx-simplifier` to resolve shapes.

## Unresolved questions
[unresolved-questions]: #unresolved-questions

- **Expression string format standardisation**: Should the expression syntax be formally specified (grammar, precedence rules, allowed functions)? Currently it is only informally documented in code comments.
- **Simplification**: Should ONNX ship a helper to reduce expressions (e.g., constant folding, identity elimination)?  This would reduce verbosity but adds complexity.
- **Consumer adoption**: Should the ONNX Python API expose utilities for parsing and evaluating `dim_param` expressions?
- **Validation**: Should `onnx.checker` validate that `dim_param` values that look like expressions are well-formed?
- **Integer vs. floor division**: Should `Ceil` division be represented differently from `Floor` division in expressions?
- **Negative and zero trip counts**: Should `LoopInferenceFunction` handle M=0 differently from M<0?

## Future possibilities
[future-possibilities]: #future-possibilities

- **Formal expression grammar**: Define a grammar for `dim_param` expression strings in the ONNX IR specification so that all conformant tools can parse them consistently.
- **Protobuf extension**: Add a dedicated `dim_expr` field to `TensorShapeProto.Dimension` with a structured expression representation (AST), replacing the string encoding.
- **Constraint propagation**: Allow users to assert that two symbolic dimensions are equal or that one is a multiple of another, enabling downstream inference to resolve more shapes statically.
- **Extended operator coverage**: Add `PartialDataPropagationFunction` to more operators (e.g., `Transpose`, `Gather`, `ScatterElements`) to propagate symbolic shape information further through shape-computation subgraphs.
- **Simplification pass**: Provide an optional pass that simplifies expression strings (e.g., folds `N - 0` to `N`, `N * 1` to `N`) to keep shapes readable.
- **Integration with onnx-simplifier and onnxruntime**: Update downstream tools to parse and evaluate `dim_param` expression strings for richer static analysis.
