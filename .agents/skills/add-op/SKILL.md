---
name: add-op
description: Add a new ONNX operator or update an existing operator to a new opset version. Use when asked to define an operator schema, register an op, add inputs/outputs/attributes to an op, or bump an op's opset version.
---

Follow the full procedure in [docs/AddNewOp.md](../../../docs/AddNewOp.md).

## Files to Modify

| Component | File |
|-----------|------|
| Schema definition | `operator-specifications/<domain>/v<N>/<op>.md` (see `operator-specifications/README.md`) |
| Type/shape inference | `shape_inference` in the spec file names a function in `onnx/defs/<domain>/utils.{h,cc}` |
| Function body (if applicable) | `function_bodies` or `function_body_builders` in the spec file |
| Reference implementation | `onnx/reference/ops/op_<lowercase_name>.py` |
| Node tests | `onnx/backend/test/case/node/<lowercase_name>.py` |
| Shape inference tests | `tests/python/shape_inference_test.py` |
| Version converter adapter (if behavior changed) | `onnx/version_converter/adapters/<name>_<from>_<to>.h` |
| Upgrade/downgrade tests | `tests/python/version_converter/automatic_upgrade_test.py` and `automatic_downgrade_test.py` |

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
        // Function body uses ONNX text format — see the onnxtxt skill for syntax/conventions.
        .FunctionBody(R"ONNX(
          {
            ...
          }
        )ONNX", FUNCTION_OPSET_VERSION));
```

## Updating an Existing Operator

1. Create `operator-specifications/<domain>/v<N>/<op>.md` with the keys that change
2. Name the type and shape inference function in `shape_inference`; the registration and the operator set are generated
3. Add a version converter adapter if behavior changed
4. Add upgrade/downgrade tests

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

Node tests belong in `onnx/backend/test/case/node/` as Python exporters with `expect(...)` calls. Backend tests discover and build these cases in memory; do not commit generated `.onnx` or `.pb` fixtures under `onnx/backend/test/data/node/`.

For test fixtures, prefer the ONNX text format via `onnx.parser` / C++ `OnnxParser`. See the [`onnxtxt`](../onnxtxt/SKILL.md) skill for the per-file recommendations table, body-subgraph idioms, the argument-order convention, and the `unk__*` materialization gotcha. (Empirical: PR #7962 cut ~58–70% of test LOC by switching.)

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
