Open Neural Network Exchange (ONNX)
=========

ONNX is an open specification that consists of the following components:

1)  A definition of an extensible computation graph model

2)  Definitions of built-in operators and standard data types

__Notes on language in this and all related documents__:

1. The use of SHOULD, MUST, MAY and so on in this document is consistent with [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

2. The use of 'list' shall denote an ordered collection of items, 'set' shall denote an unordered collection of unique elements, and 'bag' an unordered collection of possibly non-unique elements.

Extensible computation graph model
----------------------------------

ONNX specifies the portable, serialized format of the computation graph. It may not be the form a framework chooses to use and manipulate. For example, a framework may keep the graph in memory in a format that it finds more efficient to manipulate for optimization passes. We make use of protobuf2 (with oneof, added in protobuf 2.6.1) for the serialized format.

### Model

The top-level container object in an ONNX model file is a ModelProto object that combines a computation graph with the following metadata properties.

#### Model Metadata

|Name|Type|Format|Description|
|----|----|------|-----------|
|ir_version|int64||The version of the IR format specification|
|opset_import|||Custom operator sets imported by the model|
|domain|string|Valid DNS name|A namespace for the model, following the style of package names, that is, reverse DNS domain name.|
|model_version|int64||The version of the model|
|producer_name|string||Name of framework/tool that generated this model|
|producer_version|string||Version of the framework/tool that generated this model|

See [versioning documentation](Versioning.md) for more information about the versioning policies.
Further metadata may be added to a model via its `metadata_props` field (as described further in the Extensibility section below).

### Graphs

Each computation dataflow graph is structured as a list of nodes that form a graph. Each node is a call to an operator. Nodes have zero or more inputs, one or more outputs, and zero or more attribute-value pairs. Additionally, a valid ONNX graph must fulfill the following requirements:

- The graph MUST be free of cycles.
- The graph MUST be in SSA (static single assignment) form, meaning outputs of all nodes are unique.
- The nodes list MUST be in topologically sorted order, meaning if input of node `N2` is the output of node `N1`, `N2` must appear after `N1` in the nodes list.

#### Model Graph Metadata

The following describes the metadata properties of a model graph:

|Name|Type|Format|Description|
|----|----|------|-----------|
|name|string|Valid C identifier|A name for the model.|
|doc_string|string|Free form|A human-readable documentation string intended to summarize the purpose of the model. Markdown is allowed.|

#### Names Within a Graph

Names are organized into separate namespaces, and must be unique within a namespace. The namespaces include the following:
 - `Node`: names that identify specific nodes in the graph, but not necessarily any particular input or output of the node.
 - `Graph`: names that identify graphs in the protobuf.
 - `Attribute`: names that identify attribute names for extra attributes that are passed to operators.
 - `Operator`: names that identify particular operators.
 - `Tensor`: names that identify intermediate tensor values flowing through the computation of a graph.

All names MUST adhere to C identifier syntax rules.

#### Nodes

Each computation node consists of a name, the identifier of the operator to be invoked, a list of named inputs and outputs, and a list of attribute-value pairs.

Edges in the computation graph are established by outputs of one node being referenced by name in the inputs of a subsequent node.

Names of tensor values flowing through the graph are unique - a particular name can reference either a graph input or an output of a single node. Reuse of names for several outputs is not allowed.

The list of nodes defining the top-level computation graph MUST be ordered topologically \- that is, if node K follows node N in the graph, none of the data inputs of N may refer to outputs of K.

#### Values

The representation distinguishes between two kinds of values: attribute values, which are statically known, and runtime values. The type of values permitted in the two cases are different. The permitted types of attribute values are indicated by the enumeration `AttributeType`, while the permitted types of runtime values are described by `TypeProto`. 

The types of the inputs and outputs of the model must be specified, including the shapes of tensors. While the ranks of input and output tensors are statically specified, the sizes of specific dimensions (axis) may be statically unknown and are indicated so using symbolic identifiers in the shape. 

#### Optional Inputs

Some Operators have inputs that are marked as optional. There are two ways to leave an optional input unspecified. The first is to simply not provide that input. However, this is not always possible - for example, if you wish to leave the fourth input unspecified, but still provide a value for the fifth input. Therefore, any input with a name of the empty string is treated as an unspecified optional input.

Built-in Operators and Standard Data Types
------------------------------------------

### Operators

See the [operator documentation](Operators.md) for details.


### Standard data types

The following data types are supported by ONNX. Additional data types can be supported by frameworks.

|Group|Name|Description|
|-----|----|-----------|
|Floating Point Types|__float16, float (32 bit), double (64 bit)__|Values adhering to the IEEE 754-2008 standard representation of floating-point data.|
|Signed Integer Types|__int8, int16,int32,int64__|Signed integers are supported for 8-64 bit widths.|
|Unsigned Integer Types|__uint8,uint16__| Unsigned integers of 8 or 16 bits are supported.|
|Complex Types|__complex64,complex128__|A complex number with either 32- or 64-bit real and imaginary parts.|
|Other|__string__|Strings represent textual data. All strings are encoded using UTF-8.|
|Other|__bool__|Boolean value represent data with only two values, typically _true_ and _false_.|
|Other|__handle__|Handles are opaque types holding a 64-bit integer.|
|Collections|__sparse and dense tensor__|Tensors are a generalization of vectors and matrices; whereas vectors have one dimension, and matrices two, tensors can have any number of dimensions, including zero. A zero-dimensional tensor is equivalent to a scalar.|

Extensibility
-------------

ONNX is expected to evolve over time and provides features that enable users to experiment and implement additions before they are added to the specifications.

### Metadata

A model allows named metadata strings to be added via its `metadata_props` field, typically for use by tools such as converters, indexers, or documentation generators. Names are not prescribed, but some name recommendations are provided for implementations that want to record such concepts.

- `author`: the name of the person(s) who authored the model
- `company`: the name of the company or organization that authored the model
- `converted_from`: if converted from a different format, the name of the source format or framework
- `license`: a human-readable name for a license, if applicable
- `license_url`: the URL where the license text is provided

### Operators

Operators may be extended via custom domain names in the `opset_import` field.
