(l-onnx-classes)=

# Protos

This structures are defined with protobuf in files `onnx/*.proto`.
It is recommended to use function in module {ref}`l-mod-onnx-helper`
to create them instead of directly instantiated them.
Every structure can be printed with function `print` and is rendered
as a json string.

## AttributeProto

This class is used to define an attribute of an operator
defined itself by a NodeProto. It is
a named attribute containing either singular float, integer, string, graph,
and tensor values, or repeated float, integer, string, graph, and tensor values.
An AttributeProto MUST contain the name field, and *only one* of the
following content fields, effectively enforcing a C/C++ union equivalent.

```{eval-rst}
.. autoclass:: onnx.AttributeProto
    :members:
```

(l-onnx-function-proto)=

## FunctionProto

This defines a function. It is not a model but can
be used to define custom operators used in a model.

```{eval-rst}
.. autoclass:: onnx.FunctionProto
    :members:
```

(l-onnx-graph-proto)=

## GraphProto

This defines a graph or a set of nodes called from a loop or a test
for example.
A graph defines the computational logic of a model and is comprised of a parameterized
list of nodes that form a directed acyclic graph based on their inputs and outputs.
This is the equivalent of the *network* or *graph* in many deep learning
frameworks.

```{eval-rst}
.. autoclass:: onnx.GraphProto
    :members:
```

(l-onnx-map-proto)=

## MapProto

This defines a map or a dictionary. It
specifies an associative table, defined by keys and values.
MapProto is formed with a repeated field of keys (of type INT8, INT16, INT32,
INT64, UINT8, UINT16, UINT32, UINT64, or STRING) and values (of type TENSOR,
SPARSE_TENSOR, SEQUENCE, or MAP). Key types and value types have to remain
the same throughout the instantiation of the MapProto.

```{eval-rst}
.. autoclass:: onnx.MapProto
    :members:
```

(l-modelproto)=

## ModelProto

This defines a model. That is the type every converting library
returns after converting a machine learned model.
ModelProto is a top-level file/container format for bundling a ML model and
associating its computation graph with metadata.
The semantics of the model are described by the associated GraphProto's.

```{eval-rst}
.. autoclass:: onnx.ModelProto
    :members:
```

(l-nodeproto)=

## NodeProto

This defines an operator. A model is a combination of
mathematical functions, each of them represented as an onnx operator,
stored in a NodeProto.
Computation graphs are made up of a DAG of nodes, which represent what is
commonly called a *layer* or *pipeline stage* in machine learning frameworks.
For example, it can be a node of type *Conv* that takes in an image, a filter
tensor and a bias tensor, and produces the convolved output.

```{eval-rst}
.. autoclass:: onnx.NodeProto
    :members:
```

(l-operatorproto)=

## OperatorProto

This class is rarely used by users.
An OperatorProto represents the immutable specification of the signature
and semantics of an operator.
Operators are declared as part of an OperatorSet, which also defines the
domain name for the set.
Operators are uniquely identified by a three part identifier
(domain, op_type, since_version) where

- *domain* is the domain of an operator set that contains this operator specification.
- *op_type* is the name of the operator as referenced by a NodeProto.op_type
- *since_version* is the version of the operator set that this operator was initially declared in.

```{eval-rst}
.. autoclass:: onnx.OperatorProto
    :members:
```

(l-operatorsetidproto)=

## OperatorSetIdProto

This is the type of attribute `opset_import` of class ModelProto.
This attribute specifies the versions of operators used in the model.
Every operator or node belongs to a domain. All operators for the same
domain share the same version.

```{eval-rst}
.. autoclass:: onnx.OperatorSetIdProto
    :members:
```

(l-operatorsetproto)=

## OperatorSetProto

An OperatorSetProto represents an immutable set of immutable operator specifications.
The domain of the set (OperatorSetProto.domain) is a reverse-DNS name
that disambiguates operator sets defined by independent entities.
The version of the set (opset_version) is a monotonically increasing
integer that indicates changes to the membership of the operator set.
Operator sets are uniquely identified by a two part identifier (domain, opset_version)
Like ModelProto, OperatorSetProto is intended as a top-level file/wire format,
and thus has the standard format headers in addition to the operator set information.

```{eval-rst}
.. autoclass:: onnx.OperatorSetProto
    :members:
```

(l-optionalproto)=

## OptionalProto

Some input or output of a model are optional. This class must
be used in this case. An instance of class OptionalProto
may contain or not an instance of type TensorProto, SparseTensorProto,
SequenceProto, MapProto and OptionalProto.

```{eval-rst}
.. autoclass:: onnx.OptionalProto
    :members:
```

(l-onnx-sequence-proto)=

## SequenceProto

This defines a dense, ordered, collection of elements that are of homogeneous types.
Sequences can be made out of tensors, maps, or sequences.
If a sequence is made out of tensors, the tensors must have the same element
type (i.e. int32). In some cases, the tensors in a sequence can have different
shapes.  Whether the tensors can have different shapes or not depends on the
type/shape associated with the corresponding `ValueInfo`. For example,
`Sequence<Tensor<float, [M,N]>` means that all tensors have same shape. However,
`Sequence<Tensor<float, [omitted,omitted]>` means they can have different
shapes (all of rank 2), where *omitted* means the corresponding dimension has
no symbolic/constant value. Finally, `Sequence<Tensor<float, omitted>>` means
that the different tensors can have different ranks, when the *shape* itself
is omitted from the tensor-type. For a more complete description, refer to
[Static tensor shapes](https://github.com/onnx/onnx/blob/main/docs/IR.md#static-tensor-shapes).

```{eval-rst}
.. autoclass:: onnx.SequenceProto
    :members:
```

(l-onnx-sparsetensor-proto)=

## SparseTensorProto

This defines a sparse tensor.
The sequence of non-default values are encoded as a tensor of shape `[NNZ]`.
The default-value is zero for numeric tensors, and empty-string for string tensors.
values must have a non-empty name present which serves as a name for SparseTensorProto
when used in sparse_initializer list.

```{eval-rst}
.. autoclass:: onnx.SparseTensorProto
    :members:
```

(l-onnx-stringstringentry-proto)=

## StringStringEntryProto

This is equivalent to a pair of strings.
This is used to store metadata in ModelProto.

```{eval-rst}
.. autoclass:: onnx.StringStringEntryProto
    :members:
```

(l-tensorproto)=

## TensorProto

This defines a tensor. A tensor is fully described with a shape
(see ShapeProto), the element type (see TypeProto), and the
elements themselves. All available types are listed in
{ref}`l-mod-onnx-mapping`.

```{eval-rst}
.. autoclass:: onnx.TensorProto
    :members:
```

(l-tensorshapeproto)=

## TensorShapeProto

This defines the shape of a tensor or a sparse tensor.
It is a list of dimensions. A dimension can be either an integer value
or a symbolic variable. A symbolic variable represents an unknown
dimension.

```{eval-rst}
.. autoclass:: onnx.TensorShapeProto
    :members:
```

(l-traininginfoproto)=

## TrainingInfoProto

TrainingInfoProto stores information for training a model.
In particular, this defines two functionalities: an initialization-step
and a training-algorithm-step. Initialization resets the model
back to its original state as if no training has been performed.
Training algorithm improves the model based on input data.
The semantics of the initialization-step is that the initializers
in ModelProto.graph and in TrainingInfoProto.algorithm are first
initialized as specified by the initializers in the graph, and then
updated by the *initialization_binding* in every instance in
ModelProto.training_info.
The field *algorithm* defines a computation graph which represents a
training algorithm's step. After the execution of a
TrainingInfoProto.algorithm, the initializers specified by *update_binding*
may be immediately updated. If the targeted training algorithm contains
consecutive update steps (such as block coordinate descent methods),
the user needs to create a TrainingInfoProto for each step.

```{eval-rst}
.. autoclass:: onnx.TrainingInfoProto
    :members:
```

(l-typeproto)=

## TypeProto

This defines a type of a tensor which consists in an element type
and a shape (ShapeProto).

```{eval-rst}
.. autoclass:: onnx.TypeProto
    :members:
```

(l-valueinfoproto)=

## ValueInfoProto

This defines a input or output type of a GraphProto.
It contains a name, a type (TypeProto), and a documentation string.

```{eval-rst}
.. autoclass:: onnx.ValueInfoProto
    :members:
```
