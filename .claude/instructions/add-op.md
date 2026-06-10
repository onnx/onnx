# Adding or Updating an ONNX Operator

Canonical guide: [`.agents/skills/add-op/SKILL.md`](../../.agents/skills/add-op/SKILL.md). Full procedure: [`docs/AddNewOp.md`](../../docs/AddNewOp.md). For tests using the ONNX text format / `onnx.parser`, see [`.agents/skills/onnxtxt/SKILL.md`](../../.agents/skills/onnxtxt/SKILL.md).

## Workflow-specific reminders

- Schema definition lives in `onnx/defs/<domain>/defs.cc`; register in `onnx/defs/operator_sets.h`.
- Updating an op: move the old schema to `<domain>/old.cc` *before* adding the new version. Add a version-converter adapter in `onnx/version_converter/adapters/<name>_<from>_<to>.h` if behavior changed, plus upgrade/downgrade tests.
- After schema changes regenerate docs: `python onnx/defs/gen_doc.py && python onnx/backend/test/stat_coverage.py`. Do not edit `docs/Operators.md` / `docs/Changelog.md` directly.

General build/lint/DCO/copyright conventions live in [`CLAUDE.md`](../../CLAUDE.md).

