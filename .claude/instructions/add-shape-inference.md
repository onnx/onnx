# Adding Type and Shape Inference for an Operator

Canonical guide: [`.agents/skills/add-shape-inference/SKILL.md`](../../.agents/skills/add-shape-inference/SKILL.md). Background: [`docs/ShapeInference.md`](../../docs/ShapeInference.md). For test fixtures using `onnx.parser`, see [`.agents/skills/onnxtxt/SKILL.md`](../../.agents/skills/onnxtxt/SKILL.md) (also covers the C++ `unk__*` materialization gotcha for free dims).

## Workflow-specific reminders

- Inference function is inline in the schema via `.TypeAndShapeInferenceFunction(...)` in `onnx/defs/<domain>/defs.cc`. Utility helpers live in `onnx/defs/shape_inference.h`.
- Always check `hasNInputShapes(ctx, n)` before accessing shapes and `has_dim_value()` before reading dim values. Leave unknown dims unset rather than failing.
- At minimum, provide rank inference; propagate symbolic dimensions (`dim_param`) when possible.
- Prefer named `static` inference functions over inline lambdas in `ONNX_OPERATOR_SET_SCHEMA` (macro expansion breaks debugger breakpoints).
- Tests: `onnx/test/shape_inference_test.py`. The `_make_graph` / `_assert_inferred` helpers fit parameterized op-version sweeps; for one-off fixtures prefer `onnx.parser.parse_model`.

General build/lint/DCO/copyright conventions live in [`CLAUDE.md`](../../CLAUDE.md).

