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

The C++ API consists of three functions

```
const std::vector<std::string> GetAvailablePasses();

ModelProto Optimize(
    const ModelProto& mp_in,
    const std::vector<std::string>& names);

ModelProto OptimizeFixed(
    const ModelProto& mp_in,
    const std::vector<std::string>& names);
```

`GetAvailablePasses()` returns a list of available passes.

`Optimize()` accepts an input `ModelProto` and a list of optimization passes to
apply, and returns a new `ModelProto` which is the result of
applying each of the named passes in sequence to the model.

`OptimizeFixed()` accepts an `ModelProto` and a list of optimization passes to apply. It then applies fixed point optimization of the passes respecting the order of the optimization passes. The strategy used by `OptimizeFixed()` is for every pass to be run as many times as needed until the graph is unchanged by the pass. We then rerun this ordered set of passes until the group of passes do not alter the graph.

## Implementing Optimization Passes
The majority of optimization passes can be implemented using the `PredicateBasedPass`, which run transforms given a pattern matching constraint. Furthermore every pass written within ONNX must formally give an annotation of what the pass does.

```
explicit PredicateBasedPass(
      PassType pass_type,
      PassEfficiency pass_efficiency,
      PassOptimizationType pass_optimization_type)
```

`PassType` is a classification of the type of pass (e.g. Nop, Fuse). `PassEfficiency` describes the fixed point behavior of the pass, if a pass is annotated as `PassEfficiency::Complete` we assume that a two repetitive applications of this pass yield the same graph as one. If the pass cannot guarantee this, mark it as `PassEfficiency::Partial`. Our last annotation `PassOptimizationType` describes what you're attempting to optimize for (e.g. Memory, Compute, Stability).

Once you've annotated your pass the two functions needed to be implemented are:
```
bool patternMatchPredicate(Node* node);
bool runTransform(Node* node, Graph& graph, NodeDestroyType& destroy_current);
```

`runTransform` will be run for every node in the graph where `patternMatchPredicate` is true.

To register your pass use
`GlobalPassRegistry::registerPass`. Optimization passes operate on an in-memory
graph representation defined in [ir.h](/onnx/common/ir.h). There are a
number of examples in the [passes](/onnx/optimizer/passes) directory.

If your pass is at all generally applicable, please consider adding it
to the core ONNX repository.
