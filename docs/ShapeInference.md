# ONNX Shape Inference

ONNX provides an optional implementation of shape inference on ONNX
graphs. This implementation covers each of the core operators, as well
as provides an interface for extensibility. Therefore, you may choose
to invoke the existing shape inference functionality on your graphs,
or to define shape inference implementations to go along with your
custom operators (or both!). Shape inference functions are stored as a
member of the OpSchema objects.

## Invoking Shape Inference

Shape inference can be invoked either via C++ or Python. The Python
API is described, with example,
[here](PythonAPIOverview.md#running-shape-inference-on-an-onnx-model).

The C++ API consists of a single function

```
shape_inference::InferShapes(
    ModelProto& m,
    const ISchemaRegistry* schema_registry);
```

The first argument is a `ModelProto` to perform shape inference on,
which is annotated in-place with shape information. The second
argument is optional.

## Implementing Shape Inference For Custom Operators

You can add a shape inference function to your operator's Schema with

```
OpSchema& Opschema::TypeAndShapeInferenceFunction(InferenceFunction inferenceFunction);
```

`InferenceFunction` is defined in
[shape_inference.h](onnx/defs/shape_inference.h), along with the core
interface struct `InferenceContext` and an assortment of helper
methods. `InferenceContext` is the core struct which is provided to
your inference function. It allows accessing information about the
operator's inputs, and also allows writing out inferred information.

To see numerous examples, search for occurences of
`TypeAndShapeInferenceFunction` in the codebase. One that is
relatively involved is the implementation for `Concat`, in
onnx/defs/tensor/defs.cc.

## Limitations

Shape inference is not guaranteed to be complete. In particular, some
dynamic behaviors block the flow of shape inference, for example a
Reshape to a dynamically-provide shape. Also, all operators are not
required to have a shape inference implementation.

Shape inference works only with constants and simple variables. It
does not support arithmetic expressions containing variables. For
example, `Concat` on tensors of shapes `(5, 2)` and `(7, 2)` can be
inferred to produce a result of shape `(12, 2)`, but `Concat` on
tensors of shapes `(5, 2)` and `(N, 2)` will simply produce `(M, 2)`,
rather than containing a representation of `N+5`. Note that differing
unknown symbolic values will be propagated, so the `M` here represents
an unknown quantity that is the same as other occurences of `M`.

These limitations are a property of the current implementation, not
fundamental constraints - if you are in need of something more
advanced, do let us know!
