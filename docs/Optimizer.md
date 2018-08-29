# ONNX Optimizer

ONNX provides a C++ library for performing arbitrary optimizations on
ONNX models, as well as a growing list of prepackaged optimization
passes. The library also provides a convenient in-memory
representation that is much more convenient to manipulate than the raw
protobuf structs, and converters to and from the protobuf format.

The primary motivation is to share work between the many ONNX backend
implementations. Not all possible optimizations can be directly
implemented on ONNX graphs - some will need additional
backend-specific information - but many can, and our aim is to provide
all such passes along with ONNX so that they can be re-used with a
single function call.

You may be interested in invoking the provided passes, or in
implementing new ones (or both).

## Invoking The Optimizer

The optimizer may be invoked either via C++ or Python. The Python API
is described, with example,
[here](PythonAPIOverview.md#optimizing-an-onnx-model).

The C++ API consists of two functions

```
const std::vector<std::string> GetAvailablePasses();

ModelProto Optimize(
    const ModelProto& mp_in,
    const std::vector<std::string>& names);
```

`GetAvailablePasses()` returns a list of available passes.

`Optimize()` accepts an input `ModelProto` and a list of optimization passes to
apply, and returns a new `ModelProto` which is the result of
applying each of the named passes in sequence to the model.

## Implementing Optimization Passes

You can implement a new optimization pass by subclassing
`OptimizePass`, and registering your new pass with
`Optimizer::registerOptimizer()`. Optimization passes operate (at your
option) either directly on the protobuf structs, or on an in-memory
graph representation defined in [ir.h](/onnx/common/ir.h). There are a
number of examples in the [passes](/onnx/optimizer/passes) directory.

If your pass is at all generally applicable, please consider adding it
to the core ONNX repository.
