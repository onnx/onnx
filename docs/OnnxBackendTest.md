<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Backend Test

## What is ONNX Backend Test

ONNX Backend Test is a Python conformance-test suite that can be applied to an
ONNX backend implementation. It runs models and nodes with known inputs and
compares the backend's results with expected outputs. Passing the included
tests demonstrates support for those cases; it does not by itself prove full
ONNX compliance.

### Relationship to the ONNX specification

> **Draft note:** The precise normative status and lifecycle of backend tests is being
> discussed in [issue #8287](https://github.com/onnx/onnx/issues/8287). This
> section is proposed wording for that discussion and will be finalized before
> this documentation change is marked ready for review.

The ONNX IR and versioned operator specifications define normative ONNX
semantics. Backend tests provide executable conformance cases and examples
derived from those specifications. They are also useful evidence when
clarifying an ambiguity, but a test does not silently override a versioned
operator specification.

A discrepancy between a test, reference implementation, and operator
specification should be reported and investigated. The resolution may correct
the test or reference implementation, clarify an ambiguity according to the
[ONNX versioning policy](Versioning.md), or introduce a new operator version
when behavior changes.

## Test categories

The backend-test runner currently loads several categories:

- **Node tests** exercise individual operators, including different attributes,
  input types, shapes, and edge cases. Their Python definitions live in
  [`onnx/backend/test/case/node`](/onnx/backend/test/case/node).
- **Simple model tests** exercise small graphs defined in
  [`onnx/backend/test/case/model`](/onnx/backend/test/case/model). Their
  serialized fixtures are stored under
  [`onnx/backend/test/data/simple`](/onnx/backend/test/data/simple).
- **Real model tests** exercise representative models. The current models are
  lightweight fixtures committed under
  [`onnx/backend/test/data/light`](/onnx/backend/test/data/light), with metadata
  under [`onnx/backend/test/data/real`](/onnx/backend/test/data/real).
- **PyTorch-converted and PyTorch-operator tests** are historical model fixtures
  retained under [`onnx/backend/test/data`](/onnx/backend/test/data).

### Node-test data access

PR [#7959](https://github.com/onnx/onnx/pull/7959) removed the generated node
`.onnx` and `.pb` artifacts from the package. Node cases are now constructed in
memory. Consumers should load them through
[`load_model_tests(kind="node")`](/onnx/backend/test/loader/__init__.py) and use
each returned `TestCase.model` and `TestCase.data_sets` instead of assuming that
`onnx/backend/test/data/node` exists.

Earlier documentation described the removed directory as the source of truth,
and downstream projects consumed its serialized layout. A supported way to
materialize the in-memory cases for file-based and non-Python consumers is
being discussed in [issue #8288](https://github.com/onnx/onnx/issues/8288).

## Contributing

### Node tests

Node tests are written in Python and NumPy. Each operator normally has a file
under [`onnx/backend/test/case/node`](/onnx/backend/test/case/node); for example,
[`add.py`](/onnx/backend/test/case/node/add.py) contains the tests for
[`Add`](/docs/Operators.md#Add). Each `expect(...)` call defines a test case.

The source of exported test functions is also embedded as example code in the
generated [Operators documentation](/docs/Operators.md). After changing node
tests, regenerate the operator coverage and documentation as described in the
[contributor guide](/CONTRIBUTING.md).

### Model tests

Small model tests can be defined with `expect(...)` under
[`onnx/backend/test/case/model`](/onnx/backend/test/case/model). The
`backend-test-tools generate-data` command serializes these cases into the
model-test data directory; review all generated changes before committing them.

The representative models under `data/light` were made small enough for the
repository by replacing large initializers with `ConstantOfShape` nodes. Adding
a large model or new generated binary fixtures affects repository and package
size and should be discussed with maintainers first. It does not require an
administrator to upload files to cloud storage.
