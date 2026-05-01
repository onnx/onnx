---
name: add-shape-inference
description: Add or update type and shape inference for an ONNX operator. Use when asked to implement TypeAndShapeInferenceFunction, propagate shapes, add shape inference tests, fix shape inference bugs, or handle broadcasting logic.
---

See also: [docs/ShapeInference.md](../../docs/ShapeInference.md)

## File Locations

| Component | File |
|-----------|------|
| Inference function | `onnx/defs/<domain>/defs.cc` (inline with schema) |
| Utility functions | `onnx/defs/shape_inference.h` |
| Tests | `onnx/test/shape_inference_test.py` |

## Type Inference vs. Shape Inference

**Type inference** (element type) is often handled automatically by type constraints. When `"T"` is shared between input and output, the framework infers output type automatically.

However, many existing ops still explicitly call `propagateElemTypeFromInputToOutput` as a best practice for robustness.

Explicit type inference logic is only needed when:
- Output type is determined by an **attribute** (e.g., `Cast`)
- Output type differs from all inputs in a way not expressible via type constraints
- The operator uses **heterogeneous** variadic inputs/outputs

### Homogeneous vs. Heterogeneous

Applies only to variadic (repeated) inputs/outputs:
- **Homogeneous** (default): All repeated arguments share the same type. Framework propagates automatically.
- **Heterogeneous**: Each argument can differ. Used by `Loop`/`Scan`. The inference method must explicitly propagate types for each argument.

## Common Patterns

### Unary Element-wise

```cpp
.TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
```

### Binary with Broadcasting

```cpp
static void InferShapeForBinaryOp(InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    if (hasNInputShapes(ctx, 2))
        bidirectionalBroadcastShapeInference(
            ctx.getInputType(0)->tensor_type().shape(),
            ctx.getInputType(1)->tensor_type().shape(),
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
}
```

### Shape-Changing Op

```cpp
static void InferShapeForTranspose(InferenceContext& ctx) {
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
}
```

## Key Utility Functions

| Function | Purpose |
|----------|---------|
| `propagateElemTypeFromInputToOutput(ctx, in, out)` | Copy element type |
| `propagateShapeFromInputToOutput(ctx, in, out)` | Copy entire shape |
| `propagateShapeAndTypeFromFirstInput(ctx)` | Both type and shape from input 0 |
| `hasNInputShapes(ctx, n)` | Check first n inputs have shapes |
| `getOutputShape(ctx, out)` | Get mutable output shape |
| `bidirectionalBroadcastShapeInference(L, R, out)` | Numpy broadcasting |
| `getRepeatedAttribute(ctx, "name", vec)` | Get repeated attr values |
| `getAttribute(ctx, "name", default)` | Get single attr value |
| `mergeInDimensionInfo(src, dst, dim_idx)` | Merge dimension info |
| `fail_shape_inference("msg")` | Throw inference error |

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
        [("X", TensorProto.FLOAT, (2, 3, 4))],
        [make_node("OpName", ["X"], ["Y"], attr_name=value)],
        [],
    )
    self._assert_inferred(
        graph,
        [make_tensor_value_info("Y", TensorProto.FLOAT, (expected_shape))],
        opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
    )
```

Cover: known shapes, partial shapes (None), rank inference, error cases, broadcasting, attribute-dependent shapes.

## Code Style: Prefer Named Functions

Define inference functions as **separate named functions** rather than inline lambdas. The macro expansion makes breakpoints on inline lambdas unreliable.

Short one-liners (e.g., `propagateShapeAndTypeFromFirstInput`) are fine as direct references.

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
