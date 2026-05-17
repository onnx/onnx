# Adding or Updating an ONNX Operator

Canonical guide: [`.agents/skills/add-op/SKILL.md`](../../.agents/skills/add-op/SKILL.md). Full procedure: [`docs/AddNewOp.md`](../../docs/AddNewOp.md).

## Critical reminders

- Schema definition lives in `onnx/defs/<domain>/defs.cc`; register in `onnx/defs/operator_sets.h`.
- Updating an op: move the old schema to `<domain>/old.cc` before adding the new version. Add a version-converter adapter if behavior changed, plus upgrade/downgrade tests.
- Do not edit generated files (`docs/Operators.md`, `docs/Changelog.md`, `*_pb2.py`) directly.
- After schema changes: `python onnx/defs/gen_doc.py && python onnx/backend/test/stat_coverage.py`.
- After proto changes: `python onnx/gen_proto.py`.
- Lint: `lintrunner -a --output oneline`. DCO: `git commit -s`.
- New Python files need `from __future__ import annotations` and the standard copyright header.
- For tests using the ONNX text format / `onnx.parser`, see [`.agents/skills/onnxtxt/SKILL.md`](../../.agents/skills/onnxtxt/SKILL.md).
