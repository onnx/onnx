# Adding Type and Shape Inference for an Operator

Canonical guide: [`.agents/skills/add-shape-inference/SKILL.md`](../../.agents/skills/add-shape-inference/SKILL.md). Background: [`docs/ShapeInference.md`](../../docs/ShapeInference.md).

## Critical reminders

- Inference function is inline in the schema via `.TypeAndShapeInferenceFunction(...)` in `onnx/defs/<domain>/defs.cc`.
- Utility helpers live in `onnx/defs/shape_inference.h` (`propagateElemTypeFromInputToOutput`, `hasNInputShapes`, `bidirectionalBroadcastShapeInference`, `getRepeatedAttribute`, `fail_shape_inference`, …).
- Always check `hasNInputShapes(ctx, n)` before accessing shapes and `has_dim_value()` before reading dim values. Leave unknown dims unset rather than failing.
- At minimum, provide rank inference; propagate symbolic dimensions (`dim_param`) when possible.
- Prefer named `static` inference functions over inline lambdas in `ONNX_OPERATOR_SET_SCHEMA` (macro expansion breaks debugger breakpoints).
- Tests: `onnx/test/shape_inference_test.py`. The `_make_graph` / `_assert_inferred` helpers fit parameterized op-version sweeps; for one-off fixtures prefer `onnx.parser.parse_model` — see [`.agents/skills/onnxtxt/SKILL.md`](../../.agents/skills/onnxtxt/SKILL.md) (includes the C++ `unk__*` materialization gotcha and the `ExpectFreeDim` helper).
- After changes: `pytest onnx/test/shape_inference_test.py -k "test_opname" -x`, then `python onnx/defs/gen_doc.py` and `lintrunner -a --output oneline`.
