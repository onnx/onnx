<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Implementing an ONNX backend

## What is an ONNX backend

An ONNX backend is an implementation capable of executing ONNX models. A
backend may interpret a model, compile or lower it to another representation,
translate it for an existing framework, execute it directly on hardware, or
combine these approaches. ONNX does not require a particular implementation
strategy.

The ONNX Python package provides an adapter interface for exposing an
implementation to ONNX's backend-test runner. This Python interface is useful
for conformance testing, but it is not itself part of the ONNX model or runtime
specification.

## Python backend adapter interface

The interface is defined in [`onnx/backend/base.py`](/onnx/backend/base.py) and
has three core concepts:

- `Device` describes a device type and optional device identifier, such as
  `CPU`, `CUDA`, or `CUDA:1`.
- `Backend` accepts an ONNX model and prepares it for execution. Its
  `run_model` and `run_node` helpers support one-off execution where the backend
  implements the corresponding methods.
- `BackendRep` is the prepared-model handle returned by `Backend.prepare`.
  Repeated calls to `BackendRep.run` execute the model with new inputs.

The adapter may wrap an implementation written in any language. Only the
Python-facing adapter needs to implement this interface.

The repository's
[`ReferenceEvaluatorBackend`](/tests/python/backend_reference_test.py) is the
canonical current example. It adapts `ReferenceEvaluator` to `Backend` and
`BackendRep`, declares CPU support, and runs the ONNX backend suite. For an
external integration, [ONNX-TensorRT's backend test](https://github.com/onnx/onnx-tensorrt/blob/main/onnx_backend_test.py)
uses the same interface and runner. Historical Caffe2, onnx-coreml, and
onnx-tensorflow integrations are no longer used as current implementation
examples because those projects or integrations are archived or unmaintained.

## Integrating ONNX Backend Test

Create a module containing the backend adapter, construct `BackendTest`, and
export its generated test cases for pytest or unittest discovery:

```python
import onnx.backend.test

from my_backend import MyBackend

pytest_plugins = ("onnx.backend.test.report",)

backend_test = onnx.backend.test.BackendTest(MyBackend, __name__)
globals().update(backend_test.enable_report().test_cases)
```

Use `include`, `exclude`, or `xfail` patterns to describe the cases supported
by the implementation. Prefer narrowly scoped patterns with comments that
explain the unsupported behavior. See the in-repository reference-backend test
for current examples.

The suite includes individual node cases, small model cases, representative
lightweight models, and retained converted-model fixtures. See
[ONNX Backend Test](OnnxBackendTest.md) for their sources, contribution process,
and the transition from serialized node fixtures to in-memory test cases.

## Coverage report

Calling `enable_report()` and loading the `onnx.backend.test.report` pytest
plugin adds a summary such as:

```text
---------- onnx coverage: ----------
Operators (passed/loaded/total): <passed>/<loaded>/<total>
------------------------------------
```

- `passed` is the number of operator types whose loaded cases passed.
- `loaded` is the number of operator types exercised by the selected cases.
- `total` is calculated from the schema registry in the installed ONNX version.

These values change as schemas and tests are added, so documentation should not
hard-code a particular operator count. The generated
[ONNX Core Test Coverage](/docs/TestCoverage.md) and
[ONNX-ML Test Coverage](/docs/TestCoverage-ml.md) reports contain the current
repository-wide coverage.

Passing the backend suite measures the implementation against the cases that
were loaded. It is not a certification of complete ONNX support. The proposed
wording for the relationship between backend tests and the normative
specification is being discussed in
[issue #8287](https://github.com/onnx/onnx/issues/8287).
