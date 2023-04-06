<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Shape Inference

ONNX provides an optional implementation of shape inference on ONNX
graphs. This implementation covers each of the core operators, as well
as provides an interface for extensibility. Therefore, you may choose
to invoke the existing shape inference functionality on your graphs,
or to define shape inference implementations to go along with your
custom operators (or both!). Shape inference functions are stored as a
member of the OpSchema objects.

In ONNX 1.10 release, symbol generation and propagation along with shape 
data propagation was added to ONNX graph level shape inference. 
Detailed proposal is [here](proposals/SymbolicShapeInfProposal.md)

## Background

Please see this [section](IR.md#static-tensor-shapes) of IR.md for a review of static tensor shapes.
In particular, a static tensor shape (represented by a `TensorShapeProto`) is distinct from
a runtime tensor shape. This feature is commonly used when the exact runtime tensor shape is
not known statically (that is, at compile time).

* A `Tensor` with an undefined `shape` field is used to represent a tensor of unknown rank.
* A `Tensor` with a defined `shape` represents a tensor of known rank.
* Each `Dimension` of a `TensorShapeProto` can have a known integer value
(represented by the `dim_value` field) or it can have an unknown value
represented by a symbolic identified (the `dim_param` field) or it
may have neither field defined (in which case it represents an anonymous
unknown value).

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
an unknown quantity that is the same as other occurrences of `M`.

These limitations are a property of the current implementation, not
fundamental constraints - if you are in need of something more
advanced, do let us know!

## Implementing Shape Inference For Operators

You can add a shape inference function to your operator's Schema with

```
OpSchema& Opschema::TypeAndShapeInferenceFunction(InferenceFunction inferenceFunction);
```

`InferenceFunction` is defined in
[shape_inference.h](/onnx/defs/shape_inference.h), along with the core
interface struct `InferenceContext` and an assortment of helper
methods. `InferenceContext` is the core struct which is provided to
your inference function. It allows accessing information about the
operator's inputs, and also allows writing out inferred information.

To see numerous examples, search for occurrences of
`TypeAndShapeInferenceFunction` in the codebase. One that is
relatively involved is the implementation for `Concat`, in
onnx/defs/tensor/defs.cc.

Please note the following points when implementing the shape-inference method for
operators to avoid common errors:

* Before accessing the `shape` of any input, the code must check that
the shape is available. If unavailable, it should be treated as a dynamic
tensor whose rank is unknown and handled appropriately. Usually, the
shape-inference logic is guarded by a call to `hasInputShape` or
`hasNInputShapes`.

* Before accessing the `dim_value` or `dim_param` of any dimension, the
code must check if these fields have a value. In particular, the code must
handle the possibility that the dimension may not have a statically
known value.

There are several utility functions in [shape_inference.h](/onnx/defs/shape_inference.h)
to handle various common situations.

* Use `checkInputRank` for inputs that must have a fixed rank. (See the
inference for `RoiAlign` as an example.)

* `unifyInputDim` and `unifyDim` and `updateOutputShape` can be used
when multiple input dims are expected to be the same, and when input
dimensions are propagated to specific output dimensions. (See the inference
for `RoiAlign` for an example.)

* Overloaded operators `*` and `/` can be used on symbolic dimensions when output
dimensions are computed from input dimensions using arithmetic. (See the inference
for `SpaceToDepth` for an example.)

These utilities handle missing shapes and dimensions safely.

_Example_: Consider a simple matrix-multiplication op that expects inputs of shape
`[M,K]` and `[K,N]` and returns an output of shape `[M,N]`. This can be coded
up as below:
```cpp
   // Check that input 0 has rank 2 (if its rank is known).
   checkInputRank(ctx, 0, 2);
   // Check that input 1 has rank 2 (if its rank is known).
   checkInputRank(ctx, 1, 2);
   Dim M, K, N;
   // Check various dimensions, handling missing dimensions/shapes safely.
   unifyInputDim(ctx, 0, 0, M);
   unifyInputDim(ctx, 0, 1, K);
   unifyInputDim(ctx, 1, 0, K);
   unifyInputDim(ctx, 1, 1, N);
   updateOutputShape(ctx, 0, {M. N});
```

