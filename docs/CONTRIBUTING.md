<!--- SPDX-License-Identifier: Apache-2.0 -->

# Development

To build ONNX from source please follow the instructions listed [here](https://github.com/onnx/onnx#build-onnx-from-source).

Then, after you have made changes to Python and C++ files:

- `Python files`: the changes are effective immediately in your installation. You don't need to install these again.
- `C++ files`: you need to install these again to trigger the native extension build.

Assuming build succeed in the initial step, simply running
```
pip install -e .
```
from onnx root dir should work.

## Folder structure

- `onnx/`: the main folder that all code lies under
  - `onnx.proto`: the protobuf that contains all the structures
  - `checker.py`: a utility to check whether a serialized ONNX proto is legal
  - `shape_inference.py`: a utility to infer types and shapes for ONNX models
  - `version_converter.py`: a utility to upgrade or downgrade version for ONNX models
  - `parser.py`: a utility to create an ONNX model or graph from a textual representation
  - `hub.py`: a utility for downloading models from [ONNX Model Zoo](https://github.com/onnx/models)
  - `compose.py`: a utility to merge ONNX models
  - `helper.py`: tools for graph operation
  - `defs/`: a subfolder that defines the ONNX operators
  - `test/`: test files

## Generated operator documentation

[Operator docs in Operators.md](Operators.md) are automatically generated based on C++ operator definitions and backend Python snippets. To refresh these docs, run the following commands from the repo root and commit the results. Note `ONNX_ML=0` updates Operators.md whereas `ONNX_ML=1` updates Operators-ml.md:

```
set ONNX_ML=0
pip install setup.py
python onnx/defs/gen_doc.py
```

## Adding a new operator

ONNX is an open standard, and we encourage developers to contribute high
quality operators to ONNX specification.
Before proposing a new operator, please read [the tutorial](AddNewOp.md).

# Code style

We use flake8, mypy, and clang-format for checking code format.
*Note: You'll find the versions of these tools in `setup.py`.*
You can run these checks by:

```
pip install -e .[lint]

./tools/style.sh
```

# Testing

ONNX uses [pytest](https://docs.pytest.org) as a test driver. To run tests, you'll first need to install pytest:

```
pip install pytest nbval
```

After installing pytest, run from the root of the repo:

```
pytest
```

to begin the tests.

You'll need to regenerate test coverage too, by running this command from the root of the repo:

```
python onnx\backend\test\stat_coverage.py
```

## Cpp tests (googletest)

Some functionalities are tested with googletest. Those tests are listed in `test/cpp`, and include tests for shape inference, data propagation, parser, and others.

To run them, first build ONNX with `-DONNX_BUILD_TESTS=1` or `ONNX_BUILD_TESTS=1 pip install -e .` and then run `.setuptools-cmake-build/onnx_gtests`.

# Static typing (mypy)

We use [mypy](http://mypy-lang.org/) to run static type checks on the onnx code base. To check that your code passes, you'll first need to install the mypy type checker. If you're using python 3, call from your onnx source folder:

```
pip install -e .[lint]
```

*Note: You'll find the version we're currently using in `setup.py`.*

After having installed mypy, you can run the type checks:

```
python setup.py typecheck
```
# CI Pipelines

Every PR needs to pass CIs before merge. CI pipelines details are [here](CIPipelines.md).

# Other developer documentation

* [How to implement ONNX backend (ONNX to something converter)](ImplementingAnOnnxBackend.md)
* [Backend test infrastructure and how to add tests](OnnxBackendTest.md)

# License

[Apache License v2.0](/LICENSE)

# Code of Conduct

[ONNX Open Source Code of Conduct](http://onnx.ai/codeofconduct.html)
