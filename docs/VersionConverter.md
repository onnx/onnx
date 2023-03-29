<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Version Converter

ONNX provides a library for converting ONNX models between different
opset versions. The primary motivation is to improve backwards compatibility of ONNX
models without having to strengthen the spec for ONNX backends.  This
allows backend developers to offer support for a particular opset version
and for users to write or export models to a particular opset version but
run in an environment with a different opset version. Implementation wise, the library leverages the in-memory representation that is much more convenient to manipulate than the raw protobuf structs, and converters to and from the protobuf format which were developed for the ONNX Optimizer.

You may be interested in invoking the provided op-specific adapters, or in
implementing new ones (or both). Default adapters only work in the default
domain, but can be generalized to work cross-domain or utilizing new
conversion methods, dependent on the nature of relevant breaking changes.

## Invoking The Version Converter

The version converter may be invoked either via C++ or Python.

The Python API
is described, with example,
[here](PythonAPIOverview.md#converting-version-of-an-onnx-model-within-default-domain-aionnx).

The C++ API consists of a single function

```
ModelProto ConvertVersion(
    const ModelProto& mp_in,
    const OpSetID& initial_version,
    const OpSetID& target_version);
```

which accepts an input `ModelProto`, the initial opset version of the model,
and the target opset version, and which returns a new `ModelProto` which
is the result of apply all relevant adapters between initial_version and
target_version. For a list of available passes, see
[convert.h](/onnx/version_converter/convert.h).

## Implementing Adapters

You can implement a new adapter by subclassing `Adapter`, and registering
your new adapter with `VersionConverter::registerAdapter()`. Adapters operate
on an in-memory graph representation defined in [ir.h](/onnx/common/ir.h).
There are a number of examples in the [adapters](/onnx/version_converter/adapters)
directory.  Please ensure that all adapters convert from opset version i to i + 1
or i - 1, i.e. from Version 6 to Version 5 or vice versa, even if the 2 versions
being converted between are Version 1 and Version 6.

If your adapter applies in the default domain, please consider adding it
to the core ONNX repository
