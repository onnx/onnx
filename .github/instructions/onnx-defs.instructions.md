---
applyTo: "onnx/defs/**"
---

# ONNX `onnx/defs/` Quick Reference

When working in `onnx/defs/**`, follow the relevant agent skill — these contain the full, canonical guidance:

| Task | Skill |
|---|---|
| Add a new op or bump an opset | [`.agents/skills/add-op/SKILL.md`](../../.agents/skills/add-op/SKILL.md) |
| Implement type/shape inference | [`.agents/skills/add-shape-inference/SKILL.md`](../../.agents/skills/add-shape-inference/SKILL.md) |
| Add a function body decomposition | [`.agents/skills/add-function-body/SKILL.md`](../../.agents/skills/add-function-body/SKILL.md) |
| Write or interpret ONNX text format / `.FunctionBody(R"ONNX(...)")` / `onnx.parser.parse_model` | [`.agents/skills/onnxtxt/SKILL.md`](../../.agents/skills/onnxtxt/SKILL.md) |

## Critical reminders (apply to all `onnx/defs/` work)

- **Do not edit generated files** directly: `docs/Operators.md`, `docs/Changelog.md`, `docs/TestCoverage.md`, `*_pb2.py`, `onnx-*.in.proto`-derived `.proto` files. Edit the source, then regenerate.
- **After schema changes:** `python onnx/defs/gen_doc.py && python onnx/backend/test/stat_coverage.py`.
- **After proto changes:** `python onnx/gen_proto.py` (edit the `.in.proto` files, not the `.proto` files).
- **Lint before pushing:** `lintrunner -a --output oneline`.
- **DCO sign-off** on every commit: `git commit -s`.
- **New Python files** need `from __future__ import annotations` and the standard copyright header (`# Copyright (c) ONNX Project Contributors` + `# SPDX-License-Identifier: Apache-2.0`).
- **Updating an op:** move the old schema to `<domain>/old.cc` before adding the new version to `defs.cc`; update `onnx/defs/operator_sets.h`; add upgrade/downgrade tests.
- **Prefer named functions** over inline lambdas in `ONNX_OPERATOR_SET_SCHEMA` (macro expansion makes breakpoints on lambdas unreliable).

For the full procedures (file locations, registration patterns, common idioms, test recipes), open the skills above.
