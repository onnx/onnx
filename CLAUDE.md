<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# CLAUDE.md — ONNX Project Guide

ONNX (Open Neural Network Exchange) — open-source standard format for AI models. Python + C++ codebase using protobuf for serialization. Builds and runs on **Linux, macOS, and Windows** — keep all three platforms in mind when making changes.

Also follow the shared AI assistant guidelines in `.github/copilot-instructions.md`.

Before committing, pushing, filing an issue, or opening a pull request, review [CONTRIBUTING.md](CONTRIBUTING.md) — it defines the PR process, branch/CI expectations, and coding style. When writing up a bug report or evaluating its severity, check [SECURITY.md](SECURITY.md)'s disclosure policy first: easily-discovered bugs (found with widely available tooling) are fine as a normal public issue or PR, but a non-trivial security vulnerability must be reported privately via GitHub Security Advisories, not a public issue.

## Project Norms

- Follow the [ONNX Code of Conduct](https://onnx.ai/codeofconduct.html). All generated code, comments, commit messages, and PR descriptions must be professional, welcoming, and free of hostile, discriminatory, or demeaning language.
- ONNX is an **open standard** — changes to operator definitions, proto schemas, or the IR spec affect the entire ML ecosystem. Be conservative and deliberate with spec-level changes.
- Stay **vendor-neutral**. Do not favor any specific framework, runtime, or hardware in code or comments.
- Preserve **backward compatibility**. Breaking changes to the spec or public API have outsized impact across the ecosystem.
- Match existing code patterns and conventions — read surrounding code before making changes.
- Keep PRs focused. Do not bundle unrelated changes or refactor code outside the scope of the task.
- New operators must follow the process in `docs/AddNewOp.md`.
- Avoid adding new dependencies. A proposal for one must provide a compelling justification, explain why the standard library and existing dependencies are insufficient, and assess its maintenance, security, packaging, and cross-platform impact. Its license must be compatible with ONNX's Apache-2.0 distribution, and maintainers must approve the dependency before it is added.

## Build

```bash
pip install -e . -v                        # Development install
ONNX_BUILD_TESTS=1 pip install -e . -v     # With C++ tests
```

On Windows PowerShell, set `$env:ONNX_BUILD_TESTS = "1"` before running
`pip install -e . -v` to build the C++ tests.

If [pixi](https://pixi.prefix.dev/latest/) is available in your environment, `pixi run install` (and `pixi run pytest`, `pixi run gtest`, `pixi run lint`, `pixi run gen-all`) is the preferred, more reproducible way to build, test, lint, and regenerate files — see `pixi.toml` for the full task list. Fall back to the plain commands below when pixi isn't available.

Pure Python changes take effect immediately in editable installs. C++ changes require rebuild.

## Testing

```bash
pytest                                      # All Python tests

# C++ tests (build with ONNX_BUILD_TESTS=1 first)
# Linux:
LD_LIBRARY_PATH="./.setuptools-cmake-build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" .setuptools-cmake-build/onnx_gtests
# macOS:
DYLD_LIBRARY_PATH="./.setuptools-cmake-build${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" .setuptools-cmake-build/onnx_gtests
# Windows:
.setuptools-cmake-build\Release\onnx_gtests.exe
```

Tests live in `tests/` with `*_test.py` naming.

## Linting

With Pixi, run the same pre-commit hooks used by CI:

```bash
pixi run lint
```

Without Pixi, use `lintrunner`:

```bash
pip install lintrunner lintrunner-adapters
lintrunner init    # First-time setup
lintrunner         # Lint changed files
lintrunner -a      # Auto-fix
```

`lintrunner` runs ruff, mypy, clang-format, editorconfig-checker, and a namespace checker. **The applicable lint command must pass with no errors before a coding task is considered complete.**

## Code Conventions

- All Python files require `from __future__ import annotations`
- No relative imports — use absolute imports from `onnx`
- Copyright header on all files: `# Copyright (c) ONNX Project Contributors` + `# SPDX-License-Identifier: Apache-2.0`
- DCO sign-off required on all commits (`git commit -s`)

## Auto-Generated Files (Do Not Edit)

Edit the source, then regenerate. CI verifies these are up to date.

| Generated files | Source of truth | Regenerate with |
|---|---|---|
| `docs/Operators*.md`, `docs/Changelog*.md` | Op schemas in `onnx/defs/` | `python onnx/defs/gen_doc.py` |
| `docs/TestCoverage*.md` | Backend test cases in `onnx/backend/test/` | `python onnx/backend/test/stat_coverage.py` |
| Generated `.proto` and `.proto3` variants in `onnx/` | `onnx/*.in.proto` | `python onnx/gen_proto.py -l` and `python onnx/gen_proto.py -l --ml` |

Prefer `pixi run gen-all` to run all generators. After operator-schema changes,
regenerate the operator docs and test-coverage report. After protobuf changes, edit
the `.in.proto` sources, **not** generated `.proto` files, and regenerate both the
default and ML variants.

## C++/Python Boundary

Core validation (`checker`), shape inference, and version conversion are C++ exposed via nanobind (`onnx_cpp2py_export/`). Operator schemas are defined in C++ under `onnx/defs/`. Helper utilities, reference implementation, parser, and compose are pure Python.

**ONNX_ML flag** (on by default): controls traditional ML types (sequences, maps, sparse tensors). When enabled, builds use `onnx-ml.in.proto` instead of `onnx.in.proto`.

Build artifacts go to `.setuptools-cmake-build/`.

## PR descriptions

Follow the structure of [`.github/pull_request_template.md`](.github/pull_request_template.md).
Describe what the change does and why. Do not list the tests, lint runs, or build commands you ran:
CI runs the tests and linters on every PR, so such lists add length without information.
Mention testing only if it is not covered by CI (e.g., manual verification on a specific platform or with a downstream
runtime).
Include the checklist from the template verbatim and leave the box unchecked;
only the human author may check it.
