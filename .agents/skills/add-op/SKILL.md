---
name: add-op
description: Add a new ONNX operator or update an existing operator to a new opset version. Use when asked to define an operator schema, register an op, add inputs/outputs/attributes to an op, move an op to old.cc, or bump an op's opset version.
---

Follow the full procedure in [docs/AddNewOp.md](../../../docs/AddNewOp.md).

## Files to Modify

| Component | File |
|-----------|------|
| Schema definition | `onnx/defs/<domain>/defs.cc` |
| Operator set registration | `onnx/defs/operator_sets.h` |
| Type/shape inference | Inline in schema via `.TypeAndShapeInferenceFunction(...)` |
| Function body (if applicable) | Inline in schema via `.FunctionBody(...)` |
| Reference implementation | `onnx/reference/ops/op_<lowercase_name>.py` |
| Node tests | `onnx/backend/test/case/node/<lowercase_name>.py` |
| Shape inference tests | `onnx/test/shape_inference_test.py` |
| Upgrade/downgrade tests | `onnx/test/version_converter/automatic_upgrade_test.py` and `automatic_downgrade_test.py` |

Domain subdirectories under `onnx/defs/`: `math/`, `nn/`, `tensor/`, `logical/`, `reduction/`, `rnn/`, `sequence/`, `image/`, `text/`, `quantization/`, `controlflow/`, `optional/`, `traditionalml/`, `training/`

## Schema Registration Pattern

```cpp
ONNX_OPERATOR_SET_SCHEMA(
    OperatorName,
    OPSET_VERSION,
    OpSchema()
        .SetDoc(OperatorName_verN_doc)
        .Input(0, "X", "Description", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "Y", "Description", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Attr("attr_name", "Description", AttributeProto::FLOAT, default_value)
        .TypeConstraint("T", {"tensor(float)", "tensor(double)", ...}, "Description")
        .TypeAndShapeInferenceFunction(InferShapeForOperatorName)
        .FunctionBody(R"ONNX(
          {
            ...
          }
        )ONNX", FUNCTION_OPSET_VERSION));
```

## Updating an Existing Operator

1. Move current schema from `defs.cc` to `old.cc` in the same domain directory
2. Create new version in `defs.cc` with the new opset version number
3. Update `onnx/defs/operator_sets.h`
4. Add a version converter adapter if behavior changed
5. Add upgrade/downgrade tests

### Avoiding duplication between defs.cc and old.cc

When moving a schema to `old.cc`, avoid significant code/documentation duplication:

- Extract common logic (doc strings, type constraint lists, shape inference helpers) into shared functions in the domain's header or utils file.
- Use parameterized helpers when versions differ only slightly (e.g., expanded type list, additional optional input).
- Some duplication is acceptable when sharing would create overly complicated logic. Prefer clarity over DRY when the alternative makes either version harder to understand independently.

## Code Style: Prefer Named Functions

Define shape inference functions and function body builders as **separate named functions** rather than inline lambdas within `ONNX_OPERATOR_SET_SCHEMA`. The macro expansion makes setting breakpoints on inline lambdas unreliable in debuggers.

```cpp
// PREFERRED: named function — easy to set breakpoints
static void InferShapeForMyOp(InferenceContext& ctx) {
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    // ...
}

ONNX_OPERATOR_SET_SCHEMA(
    MyOp, 21,
    OpSchema()
        // ...
        .TypeAndShapeInferenceFunction(InferShapeForMyOp));
```

## Writing Tests

For writing tests with the ONNX text format (`onnx.parser.parse_model`, `parse_graph`, body subgraphs, argument-order conventions, the C++ `OnnxParser`), see the [`onnxtxt`](../onnxtxt/SKILL.md) skill. Per-file recommendations:

| Test file | Recommendation |
|-----------|----------------|
| `onnx/test/shape_inference_test.py`, `onnx/test/reference_evaluator_test.py` | Use `onnx.parser.parse_model`. |
| `onnx/backend/test/case/node/<op>.py` | Keep the outer `helper.make_node` + `expect(...)` (it drives data generation). For ops with a graph attribute (`Scan`, `Loop`, `If`, …), build the body with `onnx.parser.parse_graph`. |
| `onnx/test/cpp/shape_inference_test.cc` | Use the C++ `OnnxParser` (`onnx/defs/parser.h`): parse text, then `shape_inference::InferShapes`. |
| `onnx/test/version_converter/automatic_upgrade_test.py` and similar harnesses | Keep the established `_test_op_upgrade` style — do not rewrite. |

Empirical: PR #7962 (ScanVarLen) cut ~58–70% of test LOC by switching.

## After Making Changes

```bash
python onnx/defs/gen_doc.py
python onnx/backend/test/stat_coverage.py
python onnx/gen_proto.py  # only if proto changed
lintrunner -a --output oneline
```

## References

- [docs/AddNewOp.md](../../../docs/AddNewOp.md) — Full procedure for adding/updating ops
- [docs/AddFunctionBody.md](../../../docs/AddFunctionBody.md) — Function body guide
- [docs/ShapeInference.md](../../../docs/ShapeInference.md) — Shape inference guide
- [../onnxtxt/SKILL.md](../onnxtxt/SKILL.md) — ONNX text format ("onnxtxt") for tests and function bodies
- [references/node-test-pattern.md](references/node-test-pattern.md) — Node test example
- [references/reference-impl-pattern.md](references/reference-impl-pattern.md) — Reference implementation example
