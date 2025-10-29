# Copilot Instructions for ONNX

## Repository Overview

[Open Neural Network Exchange (ONNX)](https://onnx.ai) is an open ecosystem that empowers AI developers to choose the right tools as their project evolves. This repository contains:
- The ONNX specification and schema (protobuf definitions)
- Python bindings and utilities for working with ONNX models
- C++ implementation for model validation, shape inference, and optimization
- Reference implementations and test infrastructure

The project aims for reliability, extensibility, and broad compatibility across ML frameworks and hardware.

## Project Structure

- `onnx/`: Main source code directory
  - `onnx.proto`, `onnx-ml.proto`: Protocol buffer definitions for ONNX IR
  - `checker.py`, `checker.cc`: Model and graph validation utilities
  - `shape_inference.py`: Type and shape inference for ONNX models
  - `version_converter.py`: Version upgrade/downgrade utilities
  - `parser.py`: Text-to-model conversion utility
  - `helper.py`: Graph manipulation tools
  - `defs/`: Operator definitions (C++ and Python)
  - `backend/`: Backend test infrastructure
  - `test/`: Python test files
  - `reference/`: Reference implementation
- `docs/`: Documentation, operator specs, and guides
- `cmake/`: CMake build configuration
- `tools/`: Development and release tools
- `workflow_scripts/`: CI/CD helper scripts
- `.github/`: GitHub configuration, workflows, and issue templates

## Contribution Guidelines

### Coding Style

- **Python**: Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- **C++**: Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- Avoid hardcoding the `onnx` namespace in C++ code to allow static linking with symbol hiding
- Use clear, descriptive variable and function names
- Add comments for complex logic, but prefer self-documenting code

### Linting

We use `lintrunner` to manage multiple linters configured in .lintrunner.toml:

```sh
# Install linting tools
pip install lintrunner lintrunner-adapters
lintrunner init

# Run linters on changed files
lintrunner

# Display all lints and apply fixes
lintrunner -a

# Apply fixes only (faster)
lintrunner f
```

Linters include: Ruff (Python linting and formatting), Mypy (type checking), clang-format (C++ formatting), editorconfig-checker, and custom checks for namespace usage. See .lintrunner.toml for the complete configuration.

### Building ONNX

**Standard build from source:**

```sh
python -m pip install --quiet --upgrade pip setuptools wheel
export ONNX_BUILD_TESTS=0
export ONNX_ML=1
python -m pip install -e . -v
```

**For development (with editable install):**

```sh
pip install -e . -v
```

- Changes to Python files are immediately effective
- Changes to C++ files require rebuilding: `pip install -e . -v`

**To build with C++ tests (googletest):**

```sh
export ONNX_BUILD_TESTS=1
pip install -e . -v
export LD_LIBRARY_PATH="./.setuptools-cmake-build/:$LD_LIBRARY_PATH"
.setuptools-cmake-build/onnx_gtests
```

### Testing

**Python tests** (using pytest):

```sh
pip install pytest
pytest
```

**Regenerate test coverage:**

```sh
python onnx/backend/test/stat_coverage.py
```

**C++ tests** are in `test/cpp` and cover shape inference, data propagation, and parsing.

### Documentation

- Operator documentation (`Operators.md`, `Operators-ml.md`) is auto-generated from C++ definitions
- To regenerate docs:

```sh
export ONNX_ML=1
pip install -e . -v
python onnx/defs/gen_doc.py
```

## Technology Stack

- **Languages**: Python 3.10+, C++17
- **Build System**: CMake, setuptools
- **Core Dependencies**: protobuf >= 4.25.1
- **Testing**: pytest, googletest
- **Linting**: ruff, mypy, clang-format, editorconfig-checker
- **Python Packaging**: Uses PEP 517 build backend
- **ABI**: Provides abi3-compatible wheels for Python 3.12+

## Task Handling and Behavior

### Before Making Changes

1. Review relevant documentation in `docs/` directory
2. Check existing tests to understand expected behavior
3. Run linters and tests to establish baseline
4. For new operators, read `docs/AddNewOp.md` first

### Making Changes

- **Small, focused changes**: Prefer incremental PRs over large refactors
- **Test coverage**: Add or update tests for all changes
- **Operator changes**: Update both C++ definitions and Python tests
- **Breaking changes**: Discuss in issues first, consider version impact
- **Documentation**: Update docs when changing public APIs or operators

### Commit Messages

- Use the `-s` flag for DCO sign-off: `git commit -s -m "message"`
- All commits must be signed off (not just the PR)
- Use clear, descriptive commit messages

### CI Requirements

All PRs must pass:
- Linting checks (ruff, mypy, clang-format)
- Python tests (pytest)
- C++ tests (googletest)
- Documentation generation
- DCO check (Developer Certificate of Origin)

## Security and Compliance

- **REUSE compliant**: All files must have SPDX license headers
- **No hardcoded credentials**: Never commit secrets or API keys
- **External data**: Use `external_data_helper.py` for large model tensors
- **Input validation**: Always validate model inputs in checker functions
- **License**: Apache-2.0 - ensure all contributions are compatible

## Common Patterns and Conventions

- **Protobuf changes**: Modify `.in.proto` files (e.g., `onnx.in.proto`, `onnx-data.in.proto`) and regenerate with `onnx/gen_proto.py`
- **Operator versioning**: Follow semantic versioning for operator changes
- **Type annotations**: Use type hints in Python code; mypy checks are enforced
- **Error handling**: Provide clear, actionable error messages
- **Backward compatibility**: Maintain compatibility with existing models when possible

## Adding New Operators

Follow the comprehensive guide in `docs/AddNewOp.md`. Key steps:
1. Propose the operator in a GitHub issue first
2. Add operator schema in C++ (`onnx/defs/`)
3. Add shape inference function
4. Add type constraints and documentation
5. Add Python bindings
6. Add test cases
7. Regenerate operator documentation

## Resources

- [ONNX Website](https://onnx.ai)
- [Python API Documentation](https://onnx.ai/onnx/)
- [IR Specification](https://github.com/onnx/onnx/blob/main/docs/IR.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Installation Guide](../INSTALL.md)
