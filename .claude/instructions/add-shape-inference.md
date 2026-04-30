# Adding Type and Shape Inference for an Operator

Type and shape inference allows ONNX tools to statically determine output types and shapes. Every operator should have a `TypeAndShapeInferenceFunction`.

See also: `docs/ShapeInference.md`

## File Locations

| Component | File |
|-----------|------|
| Inference function | `onnx/defs/<domain>/defs.cc` (inline with schema) |
| Utility functions | `onnx/defs/shape_inference.h` |
| Tests | `onnx/test/shape_inference_test.py` |

## API

```cpp
.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
    // 1. Propagate element type (only if not handled by type constraints)
    // 2. Compute output shape
})
```

## Type Inference vs. Shape Inference

**Type inference** (element type of outputs) is often handled automatically by the schema's type constraints. When a type constraint variable (e.g., `"T"`) is shared between an input and an output, the framework infers the output type automatically — no explicit code needed.

However, many existing ops still explicitly call `propagateElemTypeFromInputToOutput` as a best practice. This is harmless when type constraints already cover the case, and ensures correct behavior regardless of how shape inference is invoked.

Explicit type inference logic is only needed when:
- The output type is determined by an **attribute** (e.g., `Cast` where `to` sets output type)
- The output type differs from all inputs in a way not expressible via type constraints
- The operator uses **heterogeneous** variadic inputs/outputs (see below)

### Homogeneous vs. Heterogeneous variadic inputs/outputs

The homogeneous/heterogeneous flag applies only to variadic (repeated) inputs or outputs:

- **Homogeneous** (default): All repeated arguments must share the same type. The type constraint variable enforces this — the framework propagates types automatically.
- **Heterogeneous**: Each repeated argument can have a distinct type. The type constraint variable only describes the set of *allowed* types, not a shared constraint. Used by ops like `Loop` and `Scan` whose carried state can have mixed types.

When using heterogeneous variadic arguments, the `TypeAndShapeInferenceFunction` must explicitly propagate types for each individual argument.

**Shape inference** almost always requires explicit logic, since output shapes depend on input shapes, attributes, or both.

## Common Patterns

### Unary Element-wise (output = input shape)

```cpp
.TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
```

### Binary with Broadcasting

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

### Shape-Changing Op (e.g., Transpose)

```cpp
.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (!hasNInputShapes(ctx, 1)) return;

    auto input_shape = ctx.getInputType(0)->tensor_type().shape();
    int rank = input_shape.dim_size();

    std::vector<int64_t> perm;
    getRepeatedAttribute(ctx, "perm", perm);

    auto* output_shape = getOutputShape(ctx, 0);
    for (int i = 0; i < rank; ++i) {
        *output_shape->add_dim() = input_shape.dim(perm[i]);
    }
})
```

### Multi-Input (e.g., Concat)

```cpp
.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    auto numInputs = ctx.getNumInputs();
    if (!hasNInputShapes(ctx, static_cast<int>(numInputs))) return;

    auto rank = ctx.getInputType(0)->tensor_type().shape().dim_size();
    int64_t axis = getAttribute(ctx, "axis", 0);
    if (axis < 0) axis += rank;

    // Sum dimensions along concat axis, merge others
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
| `fail_shape_inference("msg")` | Throw inference error |
| `fail_type_inference("msg")` | Throw type error |

## Dimension Arithmetic

```cpp
Dim operator*(const Dim& a, const Dim& b);
Dim operator*(const Dim& a, int64_t val);
Dim operator/(const Dim& a, int64_t divisor);
Dim multiplyDims(const TensorShapeProto& shape, int from, int upto);
```

## Writing Tests

```python
# In onnx/test/shape_inference_test.py

@parameterized.expand(all_versions_for("OpName"))
def test_opname(self, _, version) -> None:
    graph = self._make_graph(
        [("X", TensorProto.FLOAT, (2, 3, 4))],  # inputs
        [make_node("OpName", ["X"], ["Y"], attr_name=value)],
        [],
    )
    self._assert_inferred(
        graph,
        [make_tensor_value_info("Y", TensorProto.FLOAT, (expected_shape))],
        opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
    )
```

### Test cases to cover:
1. Known input shapes → known output shape
2. Partial shapes (some dims unknown, use `None`)
3. Rank inference (dims unknown but rank known)
4. Error cases (invalid inputs → `fail_shape_inference`)
5. Broadcasting correctness
6. Attribute-dependent shapes

## Rules for Robust Inference

1. Always check `hasNInputShapes(ctx, n)` before accessing shapes
2. Always check `has_dim_value()` before using `dim_value()`
3. Handle unknown dimensions gracefully — leave unset, don't fail
4. At minimum provide rank inference (correct number of output dims)
5. Propagate symbolic dimensions (`dim_param`) when possible

## After Making Changes

```bash
pytest onnx/test/shape_inference_test.py -k "test_opname" -x
python onnx/defs/gen_doc.py
lintrunner -a --output oneline
```

## Common Mistakes

- Accessing shapes without `hasNInputShapes` guard
- Accessing `dim_value()` without `has_dim_value()` check
- Forgetting type inference (only doing shape)
- Failing on unknown dimensions instead of leaving unset
- Not being deterministic or having side effects

## Code Style: Prefer Named Functions

Define inference functions as **separate named functions** rather than inline lambdas within `ONNX_OPERATOR_SET_SCHEMA`. This makes it easier to set debugger breakpoints (the macro expansion makes breakpoints on inline lambdas unreliable).

```cpp
// PREFERRED: named function
static void InferShapeForMyOp(InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (!hasNInputShapes(ctx, 1)) return;
    // ...
}

ONNX_OPERATOR_SET_SCHEMA(
    MyOp, 21,
    OpSchema()
        .TypeAndShapeInferenceFunction(InferShapeForMyOp));
```

Short one-liners (e.g., `propagateShapeAndTypeFromFirstInput`) are fine as direct references.
