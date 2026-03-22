<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# CLAUDE.md — ONNX Project Guide

ONNX (Open Neural Network Exchange) — open-source standard format for AI models. Python + C++ codebase using protobuf for serialization. Builds and runs on **Linux, macOS, and Windows** — keep all three platforms in mind when making changes.

Also follow the shared AI assistant guidelines in `.github/copilot-instructions.md`.

## Project Norms

- Follow the [ONNX Code of Conduct](https://onnx.ai/codeofconduct.html). All generated code, comments, commit messages, and PR descriptions must be professional, welcoming, and free of hostile, discriminatory, or demeaning language.
- ONNX is an **open standard** — changes to operator definitions, proto schemas, or the IR spec affect the entire ML ecosystem. Be conservative and deliberate with spec-level changes.
- Stay **vendor-neutral**. Do not favor any specific framework, runtime, or hardware in code or comments.
- Preserve **backward compatibility**. Breaking changes to the spec or public API have outsized impact across the ecosystem.
- Match existing code patterns and conventions — read surrounding code before making changes.
- Keep PRs focused. Do not bundle unrelated changes or refactor code outside the scope of the task.
- New operators must follow the process in `docs/AddNewOp.md`.

## Build

```bash
pip install -e . -v                        # Development install
ONNX_BUILD_TESTS=1 pip install -e . -v     # With C++ tests
```

Pure Python changes take effect immediately in editable installs. C++ changes require rebuild.

## Testing

```bash
pytest                                      # All Python tests

# C++ tests (build with ONNX_BUILD_TESTS=1 first)
# Linux/macOS:
LD_LIBRARY_PATH=./.setuptools-cmake-build/ .setuptools-cmake-build/onnx_gtests
# Windows:
.setuptools-cmake-build\Release\onnx_gtests.exe
```

Tests live in `onnx/test/` with `*_test.py` naming.

## Linting

```bash
lintrunner init    # First-time setup
lintrunner         # Lint changed files
lintrunner -a      # Auto-fix
```

Runs ruff, mypy, clang-format, editorconfig-checker, and a namespace checker.

## Code Conventions

- All Python files require `from __future__ import annotations`
- No relative imports — use absolute imports from `onnx`
- Copyright header on all files: `# Copyright (c) ONNX Project Contributors` + `# SPDX-License-Identifier: Apache-2.0`
- DCO sign-off required on all commits (`git commit -s`)

## Auto-Generated Files (Do Not Edit)

Edit the source, then regenerate. CI verifies these are up to date.

| Generated files | Source of truth | Regenerate with |
|---|---|---|
| `docs/Operators.md`, `docs/Changelog.md`, `docs/TestCoverage.md` | Op schemas in `onnx/defs/` | `python onnx/defs/gen_doc.py` |
| `onnx/*_pb2.py`, `onnx/*_pb.h`, `onnx/onnx_data.proto` | `onnx/onnx.in.proto`, `onnx/onnx-ml.in.proto` | `python onnx/gen_proto.py` |
| `onnx/backend/test/data/` node tests | Op schemas + reference impl | `python onnx/backend/test/stat_coverage.py` |

Edit `.in.proto` files, **not** `.proto` files. When adding/changing operator schemas, run all three scripts.

## C++/Python Boundary

Core validation (`checker`), shape inference, and version conversion are C++ exposed via pybind11 (`onnx_cpp2py_export/`). Operator schemas are defined in C++ under `onnx/defs/`. Helper utilities, reference implementation, parser, and compose are pure Python.

**ONNX_ML flag** (on by default): controls traditional ML types (sequences, maps, sparse tensors). When enabled, builds use `onnx-ml.in.proto` instead of `onnx.in.proto`.

Build artifacts go to `.setuptools-cmake-build/`.
