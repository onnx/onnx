---
name: add-op
description: Add a new ONNX operator or update an existing operator to a new opset version. Use when asked to define an operator schema, register an op, add inputs/outputs/attributes to an op, move an op to old.cc, or bump an op's opset version.
---

Follow the full procedure in [docs/AddNewOp.md](../../docs/AddNewOp.md).

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

## Writing Tests: Prefer `onnx.parser` for Model Fixtures

The ONNX text format is far more compact and readable than chains of `helper.make_*` calls. Use it wherever you are hand-constructing a model or subgraph for a test. For a concrete before/after example see PR #7962 (ScanVarLen), where parser-based rewrites reduced test LOC by roughly 58–70% across three files.

| Test file | Recommendation |
|-----------|----------------|
| `onnx/test/shape_inference_test.py` | Prefer `onnx.parser.parse_model` for the model fixture. |
| `onnx/test/reference_evaluator_test.py` | Prefer `onnx.parser.parse_model`. |
| `onnx/backend/test/case/node/<op>.py` | Keep the outer `helper.make_node` + `expect(...)` (it drives data generation). When the op takes a graph attribute (`Scan`, `Loop`, `If`, …), build the body subgraph with `onnx.parser.parse_graph` instead of `helper.make_graph` + a chain of `make_node` / `make_tensor_value_info` calls. |
| `onnx/test/cpp/shape_inference_test.cc` | Prefer the C++ `OnnxParser` (`onnx/defs/parser.h`): express the model as text, `ParseModel`, then `shape_inference::InferShapes`. |
| `onnx/test/version_converter/automatic_upgrade_test.py` and similar helper-driven harnesses | Keep the established convention (e.g. `_test_op_upgrade`). Do not rewrite. |

### Text-format cheat sheet

```
<ir_version: 8, opset_import: ["" : 18]>
agraph (float[N] x, float[N] y) => (float[N] z) {
    z = Add(x, y)
}
```

Body-graph attributes (Scan / Loop / If) are embedded inline:

```
out, state_out = Scan(state, x) <num_scan_inputs=1, body = b (float[1] s, float[1] xi) => (float[1] so, float[1] xo) { so = Identity(s) xo = Identity(xi) }>
```

Convention: put **inputs `(...)` before attributes `<...>`** in call sites — `Op(inputs)<attrs>` reads more naturally than `Op<attrs>(inputs)` and matches the body of PR #7962.

### Before / after (illustrative)

```python
# Before: chain of helper.make_* calls
node = helper.make_node("Add", ["x", "y"], ["z"])
graph = helper.make_graph(
    [node], "g",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None]),
     helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
    [helper.make_tensor_value_info("z", TensorProto.FLOAT, [None])])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

# After: single text fixture
model = onnx.parser.parse_model("""
    <ir_version: 8, opset_import: ["" : 18]>
    g (float[N] x, float[N] y) => (float[N] z) { z = Add(x, y) }
""")
```

## After Making Changes

```bash
python onnx/defs/gen_doc.py
python onnx/backend/test/stat_coverage.py
python onnx/gen_proto.py  # only if proto changed
lintrunner -a --output oneline
```

## References

- [docs/AddNewOp.md](../../docs/AddNewOp.md) — Full procedure for adding/updating ops
- [docs/AddFunctionBody.md](../../docs/AddFunctionBody.md) — Function body guide
- [docs/ShapeInference.md](../../docs/ShapeInference.md) — Shape inference guide
- [references/node-test-pattern.md](references/node-test-pattern.md) — Node test example
- [references/reference-impl-pattern.md](references/reference-impl-pattern.md) — Reference implementation example
