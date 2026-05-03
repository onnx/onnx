---
applyTo: "onnx/defs/**"
---

# Adding Type and Shape Inference for an Operator

Type and shape inference allows ONNX tools to statically determine output types and shapes without running the model. Every operator should have a `TypeAndShapeInferenceFunction`.

See also: [docs/ShapeInference.md](../../docs/ShapeInference.md)

## File Locations

| Component | File |
|-----------|------|
| Inference function | `onnx/defs/<domain>/defs.cc` (inline with schema) |
| Utility functions | `onnx/defs/shape_inference.h` |
| Tests | `onnx/test/shape_inference_test.py` |

## API

The inference function is a lambda receiving an `InferenceContext&`:

```cpp
.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
    // 1. Propagate element type (only if not handled by type constraints)
    // 2. Compute output shape
})
```

## Type Inference vs. Shape Inference

**Type inference** (element type of outputs) is often handled automatically by the schema's type constraints — no explicit code needed. When a type constraint variable (e.g., `"T"`) is shared between an input and an output, the framework infers the output type from the input type automatically.

However, many existing ops still explicitly call `propagateElemTypeFromInputToOutput` as a best practice for robustness. This is harmless when type constraints already cover the case, and ensures correct behavior regardless of how shape inference is invoked.

You only need explicit type inference logic when:
- The output type is determined by an **attribute** (e.g., `Cast` where `to` attribute sets output type)
- The output type differs from all input types in a way not expressible via type constraints
- The output type depends on runtime conditions
- The operator uses **heterogeneous** variadic inputs/outputs (see below)

### Homogeneous vs. Heterogeneous variadic inputs/outputs

The homogeneous/heterogeneous flag applies only to variadic (repeated) inputs or outputs:

- **Homogeneous** (default): All repeated arguments must have the same type. The type constraint variable constrains them all to be identical — the framework enforces and propagates this automatically.
- **Heterogeneous**: Each repeated argument can have a distinct type. The type constraint variable only describes the set of *allowed* types — it does not constrain different arguments to have the same type. Ops like `Loop` and `Scan` use heterogeneous variadic inputs/outputs because their carried state can have mixed types.

When using heterogeneous variadic arguments, the operator's `TypeAndShapeInferenceFunction` must explicitly propagate types for each individual argument, since the framework cannot do it automatically.

**Shape inference** almost always requires explicit logic, since output shapes depend on input shapes, attributes, or both.

## Common Patterns

### Pattern 1: Unary Element-wise (output shape = input shape)

```cpp
.TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
```

This built-in handles both type and shape propagation for ops like Relu, Abs, Neg, etc.

### Pattern 2: Binary with Broadcasting (Add, Mul, Sub, etc.)

```cpp
.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
})
```

### Pattern 3: Shape-Changing Op (Transpose)

```cpp
.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (!hasNInputShapes(ctx, 1)) {
        return;
    }
    auto input_shape = ctx.getInputType(0)->tensor_type().shape();
    int rank = input_shape.dim_size();

    std::vector<int64_t> perm;
    getRepeatedAttribute(ctx, "perm", perm);
    // ... validate perm ...

    auto* output_shape = getOutputShape(ctx, 0);
    for (int i = 0; i < rank; ++i) {
        *output_shape->add_dim() = input_shape.dim(perm[i]);
    }
})
```

### Pattern 4: Multi-Input with Dimension Computation (Concat)

```cpp
.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    auto numInputs = ctx.getNumInputs();
    if (!hasNInputShapes(ctx, static_cast<int>(numInputs))) {
        return;
    }

    auto rank = ctx.getInputType(0)->tensor_type().shape().dim_size();
    int64_t axis = getAttribute(ctx, "axis", 0);
    if (axis < 0) axis += rank;

    auto* output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
    // Copy non-concat dims, sum concat dim
    // ...
})
```

## Key Utility Functions

| Function | Purpose |
|----------|---------|
| `propagateElemTypeFromInputToOutput(ctx, in, out)` | Copy element type |
| `propagateShapeFromInputToOutput(ctx, in, out)` | Copy entire shape |
| `propagateShapeAndTypeFromFirstInput(ctx)` | Both type and shape from input 0 |
| `hasNInputShapes(ctx, n)` | Check first n inputs have shapes |
| `hasInputShape(ctx, n)` | Check if input n has shape |
| `getOutputShape(ctx, out)` | Get mutable output shape |
| `updateOutputShape(ctx, out, shape)` | Set output shape |
| `bidirectionalBroadcastShapeInference(L, R, out)` | Numpy broadcasting |
| `multidirectionalBroadcastShapeInference(shapes, out)` | N-way broadcasting |
| `getRepeatedAttribute(ctx, "name", vec)` | Get repeated attr values |
| `getAttribute(ctx, "name", default)` | Get single attr value |
| `mergeInDimensionInfo(src, dst, dim_idx)` | Merge dimension info |
| `appendSingleDimCopiedFromInputTypeToOutputType(ctx, in, out, dim)` | Copy one dimension |
| `fail_shape_inference("msg")` | Throw inference error |
| `fail_type_inference("msg")` | Throw type error |

## Dimension Arithmetic

```cpp
// Available operators on TensorShapeProto::Dimension
Dim operator*(const Dim& a, const Dim& b);
Dim operator*(const Dim& a, int64_t val);
Dim operator/(const Dim& a, int64_t divisor);
Dim multiplyDims(const TensorShapeProto& shape, int from, int upto);
```

## Writing Shape Inference Tests

```python
# In onnx/test/shape_inference_test.py

@parameterized.expand(all_versions_for("OpName"))
def test_opname(self, _, version) -> None:
    graph = self._make_graph(
        [("X", TensorProto.FLOAT, (2, 3, 4))],  # inputs: (name, type, shape)
        [make_node("OpName", ["X"], ["Y"], attr_name=attr_value)],
        [],  # value_info
    )
    self._assert_inferred(
        graph,
        [make_tensor_value_info("Y", TensorProto.FLOAT, (expected_shape))],
        opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
    )
```

### Test Cases to Cover

1. **Basic shape propagation** — known input shapes produce known output shape
2. **Partial shapes** — some dimensions unknown (use `None` in shape tuples)
3. **Rank inference** — when exact dims are unknown but rank is known
4. **Error cases** — invalid inputs that should trigger `fail_shape_inference`
5. **Broadcasting** — verify broadcast rules are applied correctly
6. **Attribute-dependent shapes** — output shape depends on attribute values

## Rules for Robust Inference

1. **Always check shape availability** before accessing dimensions:
   ```cpp
   if (!hasNInputShapes(ctx, 1)) return;
   ```

2. **Always check dim_value availability** before using:
   ```cpp
   if (dim.has_dim_value()) { /* safe to use dim.dim_value() */ }
   ```

3. **Handle unknown dimensions gracefully** — don't fail, just leave dims unset

4. **At minimum, provide rank inference** — set the correct number of output dimensions even if values are unknown

5. **Propagate symbolic dimensions** (`dim_param`) when possible for symbolic shape inference

## After Making Changes

```bash
pytest onnx/test/shape_inference_test.py -k "test_opname" -x
python onnx/defs/gen_doc.py
lintrunner -a --output oneline
```

## Common Mistakes to Avoid

- Don't access input shapes without checking `hasNInputShapes` first
- Don't access `dim_value()` without checking `has_dim_value()` first
- Don't forget to propagate element type (type inference) in addition to shape
- Don't silently return without setting anything — at minimum set the output type
- Don't fail on unknown dimensions — leave them unset instead
- Shape inference must be deterministic and side-effect free

## Code Style: Prefer Named Functions

Define inference functions as **separate named functions** rather than inline lambdas within `ONNX_OPERATOR_SET_SCHEMA`. This makes it easier to set debugger breakpoints (the macro expansion makes breakpoints on inline lambdas unreliable) and improves readability.

```cpp
// PREFERRED: named function — easy to set breakpoints
static void InferShapeForMyOp(InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (!hasNInputShapes(ctx, 1)) return;
    // ...shape logic...
}

ONNX_OPERATOR_SET_SCHEMA(
    MyOp, 21,
    OpSchema()
        // ...
        .TypeAndShapeInferenceFunction(InferShapeForMyOp));
```

```cpp
// AVOID: inline lambda (harder to debug)
ONNX_OPERATOR_SET_SCHEMA(
    MyOp, 21,
    OpSchema()
        // ...
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
            // breakpoints here are unreliable due to macro expansion
        }));
```

Short one-liners (e.g., `propagateShapeAndTypeFromFirstInput`) are fine as direct references since they're already named functions.
