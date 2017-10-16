Open Neural Network Exchange (ONNX)
=========

ONNX is an open specification that consists of the following components:

1)  Definition of an extensible computation graph model.

2)  Definition of built-in operators and standard data types.

__Some notes on language in this and all related documents__:

1. The use of SHOULD, MUST, MAY and so on in this document is consistent with [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

2. The use of 'list' shall denote an ordered collection of items, 'set' shall denote an unordered collection of unique elements, and 'bag' an unordered colletion of possibly non-unique elements.

Extensible computation graph model
----------------------------------

ONNX specifies the portable, serialized format of the computation graph. It may not be the form a framework chooses to use and
manipulate internally. For example, a framework may keep the graph in memory in another format that it finds more efficient to
manipulate for optimization passes. We make use of protobuf2 (with oneof, added in protobuf 2.6.1) for the serialized format.

### Graphs

Each computation dataflow graph is structured as a list of nodes that form a graph, which MUST be free of cycles.
Nodes have one or more inputs and one or more outputs. Each node is a call to an operator.

#### Metadata

The following are the metadata properties of a model graph:

|Name|Type|Format|Description|
|----|----|------|-----------|
|name|string|Valid C identifier|A name for the model.|
|domain|string|Valid DNS name|A namespace for the model, following the style of package names, that is, reverse DNS domain name.|
|ir_version|int64||The version of the IR format specification|
|doc_string|string|Free form|A human-readable documentation string intended to summarize the purpose of the model. Markdown is allowed.|


#### Names Within a Graph

We organize names into separate namespaces. Names must be unique within a namespace.
The namespaces are as follows:
 - Node: These names identify specific nodes in the graph but not necessarily any particular input or output of the node.
 - Graph: These names identify graphs in the protobuf.
 - Attribute: These names identify attribute names for extra attributes that are passed to operators.
 - Operator: These names identify particular operators.
 - Tensor: These names identify intermediate tensor values flowing through the computation of a graph.

All names MUST adhere to C identifier syntax rules.

#### Nodes

Computation nodes are comprised of a name, a list of named inputs, a list of named outputs, and a list of attributes.

Edges in the computation graph are established by outputs of one node being referenced by name in the inputs of a
subsequent node.

The list of nodes defining the top-level computation graph MUST be ordered topologically; that is, if node K
follows node N in the graph, none of the data inputs of N may refer to outputs of K; further, no control input of N may refer to K.


Built-in Operators and Standard Data Types
------------------------------------------

### Operators

See [the operator documentation](Operators.md) for details


### Standard data types

The following data types are supported by ONNX. Additional data types can be supported by frameworks.

|Group|Name|Description|
|-----|----|-----------|
|Floating Point Types|__float16, float32, float64__|Values adhering to the IEEE 754-2008 standard representation of floating-point data.|
|Signed Integer Types|__int8, int16,int32,int64__|Signed integers are supported for 8-64 bit widths.|
|Unsigned Integer Types|__uint8,uint16__| Unsigned integers of 8 or 16 bits are supported.|
|Complex Types|__complex64,complex128__|A complex number with either 32- or 64-bit real and imaginary parts.|
|Other|__string__|Strings represent textual data. All strings are encoded using UTF-8.|
|Other|__bool__|Boolean value represent data with only two values, typically _true_ and _false_.|
|Other|__handle__|Handles are opaque types holding a 64-bit integer.|
|Collections|__sparse and dense tensor__|Tensors are a generalization of vectors and matrices; whereas vectors have one dimension, and matrices two, tensors can have any number of dimenstions, including zero. A zero-dimensional tensor is equivalent to a scalar.|
