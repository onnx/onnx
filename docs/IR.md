Open Neural Network Exchange (ONNX) v1
=========

__Purpose__

This document contains the normative specification of the semantics of ONNX. The .proto and .proto3 files found under the ‘onnx’ folder form the normative specification of its syntax. Commentary found in the .proto and .proto3 files are intended to improve readability of those files, but are not normative if they conflict with this document. Such conflicts should be reported as documentation bugs.

__Notes on language in this and all related documents__:

1. The use of SHOULD, MUST, MAY and so on in this document is consistent with [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

2. The use of 'list' shall denote an ordered collection of items, 'set' shall denote an unordered collection of unique elements, and 'bag' an unordered collection of possibly non-unique elements.

## Components

ONNX is an open specification that consists of the following components:

1)  A definition of an extensible computation graph model.

2)  Definitions of standard data types.

3)  Definitions of built-in operators.

Of these, #1 and #2 are covered in this document; the built-in operators are covered separately in files listed at the end of this document.

## Runtime Agnostic

ONNX does not pre-suppose or imply any particular method of runtime implementation. 

An implementation may consist of a rich runtime which interprets the model; it may be a code generator that translates the model to executable code for some target programming language in its entirety; it may be a combination of the two.

Nothing in this specification should be construed as advocating one implementation approach over any other; any comments on the inner workings of concrete implementations are to be interpreted as examples.

## Extensible computation graph model

ONNX specifies the portable, serialized format of a computation graph. It does not have to be the form a framework chooses to use and manipulate the computation internally. For example, an implementation may represent the model differently in memory if it is more efficient to manipulate during optimization passes.

An implementation MAY extend ONNX is by adding operators expressing semantics beyond the standard set of operators that all implementations MUST support.

### Models

The top-level ONNX construct is a ‘Model,’ which has the following components:

|Name|Type|Description|
|---|---|---|
|ir_version|int64|The ONNX version assumed by the model.|
|opset_import|OperatorSetId|A collection of operator set identifiers made available to the model. An implementation must support all operators in the set or reject the model.|
|producer_name|string|The name of the tool used to generate the model.|
|producer_version|string|A string representing the version of the generating tool.|
|domain|string|A reverse-DNS name to indicate the model namespace or domain, for example, org.onnx|
|model_version|int64|A version of the model itself, encoded in an integer.|
|doc_string|string|A human-readable documentation for this model. Markdown is allowed.|
|graph|Graph|The parameterized graph that is evaluated to execute the model.|
|metadata_props|map<string,string>|Named metadata values; keys should be distinct.|

The main purpose of the model structure is to associate metadata with the graph, which contains all the executable elements. The metadata is used when first reading the model file, giving an implementation the information that it needs in order to determine whether it will be able to execute the model.

The metadata is also useful to tools, such as IDEs and model galleries, which needs it for the purpose of informing humans about a given model’s purpose and characteristics.

### Optional Metadata

The 'metadata_props' field in the model is available for any kind of optional metadata that a tool or model developer chooses to place there. The following are the defined “standard” optional metadata properties of a model. 

Name|Type|Format|Description
|---|---|---|---|
model_author|string|A comma-separated list of names.|The personal name of the author(s) of the model, and/or their organizations.
model_license|string|Name or URL.|The well-known name or URL of the license under which the model is made available.


### Operator Sets

Each model MUST explicitly name the operator sets that it relies on for its functionality. Operator sets define the available operators, their version, and their status. Each model defines the imported operator sets by their domains.

Each operator set is defined in a separate document, also using protobuf as the serialization format. How operator set documents are found and identified is implementation-dependent.

The properties of an operator set are:

Name|Type|Description
|---|---|---|
magic|string|The value ‘ONXXOPSET’
ir_version|int32|The ONNX version corresponding to the operators.
ir_version_prerelease|string|The prerelease component of the SemVer of the IR.
ir_build_metadata|string|The symbolic identifier of the operator to invoke.
domain|string|The domain of the operator set. Must be unique among all sets.
opset_version|int64|The version of the set of operators. 
doc_string|string|A human-readable documentation for this set of operators. Markdown is allowed.
operator|Operator[]|The operators of this operator set.

The operator set version is a simple integer value that is monotonically increased as new versions of the operator set are published. No operator in a given operator set may have a version number greater than the operator set’s version.

### Operators

Each operator used within a graph MUST be explicitly declared by one of the operator sets imported by the model.

The properties of an operator definition are:

Name|Type|Description
|---|---|---|
op_type|string|The name of the operator, as used in graph nodes. MUST be unique within the operator set’s domain.
since_version|int64|The version of the operator set when this operator was introduced.
status|OperatorStatus|One of ‘EXPERIMENTAL’ or ‘STABLE.’
doc_string|string|A human-readable documentation string for this operator. Markdown is allowed.

The version value MUST be the same value as the operator set version when the operator was first published. Subsequent versions of the operator set MUST NOT alter the signature or semantics of the operator once published as STABLE.

The ‘status’ property indicates whether the syntax, semantics, or presence of the operator is in an experimental or stable stage. Once an operator is published as STABLE, it’s syntax and semantics MUST NOT change in subsequent versions of the operator set.

There are two distinct ways to pass information to operators – inputs and attributes. The latter are used for values that are literal constants in the graph, while the former represent graph inputs or values computed elsewhere in the graph. This distinction may be highly relevant to good performance for some implementations, while completely irrelevant to others.

### Graphs

A serialized graph is comprised of a set of metadata fields, a list of model parameters, and a list of computation nodes.

Each computation dataflow graph is structured as a topologically sorted list of nodes that form a graph, which MUST be free of cycles. Each node represents a call to an operator. Each node has zero or more inputs and one or more outputs.

Graphs have the following properties:

|Name|Type|Description|
|---|---|---|
name|string|The name of the model graph.
node|Node[]|A list of nodes, forming a partially ordered computation graph based on input/output data dependencies.
initializer|Tensor[]|A list of named tensor values, used to specify default values for some of the inputs of the graph. Each initializer value is associated with an input by name matching.
doc_string|string|A human-readable documentation for this model. Markdown is allowed.
input|ValueInfo[]|The input “parameters” of the graph, possibly initialized by a default value found in ‘initializer.’
output|ValueInfo[]|The output parameters of the graph. Once all output parameters have been written to by a graph execution, the execution is complete.
value_info|ValueInfo|__TODO: Write this description__


Each graph MUST define the names and types of its inputs and outputs, which are specified as ‘value info’ structures, having the following properties:

Name|Type|Description
|---|---|---|
name|string|The name of the value/parameter.
type|Type|The type of the value.
doc_string|string|A human-readable documentation for this value. Markdown is allowed.

Each graph MUST specify a name and a domain. Domains MUST be specified using reverse domain names as organization identifiers, the same convention that is used for naming Java packages.

Graphs SHOULD be populated with documentation strings, which MAY be interpreted using GitHub-style markdown syntax. HTML and other text-markup languages MAY NOT be used in documentation strings.

### Names Within a Graph

All names MUST adhere to C identifier syntax rules.

Names of nodes, inputs, outputs, initializers, and attributes are organized into several namespaces. Within a namespace, each name MUST be unique within that graph. 

The namespaces are:

Namespace|Description
|---|---|
Attribute|The names of attributes of an operator. Unique for each operator.
Value|The names of values – node inputs & outputs, tensor values (if named), graph inputs, outputs.
Node|The names of graph nodes.
Graph|The names of graphs within a domain.
Operator|The names of operators within a domain.
Shape|The names of tensor shape variables – scoped to the value information records of a graph, which is where shape variables occur.


### Nodes

Computation nodes are comprised of a name, a list of named inputs, a list of named outputs, and a list of attributes. 

They have the following properties:

Name|Type|Description
|---|---|---|
name|string|An optional name of the node, used for diagnostic purposes only.
input|string[]|Names of the values used by the node to propagate input values to the node operator. It must refer to either a graph input or a node output.
output|string[]|Names of the outputs used by the node to capture data from the operator invoked by the node. It either introduces a  value in the graph or refers to a graph output.
op_type|string|The symbolic identifier of the operator to invoke.
domain|string|The domain of the operator set that contains the operator named by the op_type.
attribute|Attribute[]|Named attributes, another form of operator parameterization, used for constant values rather than propagated values.
doc_string|string|A human-readable documentation for this value. Markdown is allowed.

Edges in the computation graph are established by outputs of one node being referenced by name in the inputs of a subsequent node.

Node inputs MAY also refer to graph inputs and initializers. The outputs of a given node MAY introduce new names into the graph, name a graph output, or coincide with the outputs of other nodes. Thus, using overlapping output names, two nodes MAY compute the same output value. For example, when used with the (experimental) <a href="Operators.md#If">conditional operator 'If()'</a>, it is expected that each of its two branches will compute the same set of values. Graph outputs MAY NOT be used to establish data dependency edges in the graph by being named as node inputs.

 Node dependencies MUST NOT create cycles in the computation graph. 

__[[ DESCRIBE VARARGS ]]__

The list of nodes defining the top-level computation graph MUST be ordered topologically; that is, if node K follows node N in the graph, none of the data inputs of N may refer to outputs of K.

Node attributes are used to pass literal (static) values to operators.

#### Attributes

Attributes, which represent literal constants passed to operators from nodes, have the following properties:

Name|Type|Description
|---|---|---|
name|string|The name of the attribute. Must be unique among attributes, inputs, and outputs for any given operator and node.
doc_string|string|A human-readable documentation for this value. Markdown is allowed.
type|AttributeType|The type of the attribute, determining which of the remaining fields is used to hold the value of the attribute.
f|float|A floating-point value.
i|int64|An integer value
S|byte[]|UTF-8 string
t|Tensor|A tensor value
g|Graph|A graph
floats|float[]|List of floats
ints|int64[]|List of integers
strings|byte[][]|List of strings
tensors|Tensor[]|List of tensor values
graphs|Graph[]|List of graphs

The properties ‘name’ and ‘type’ are required on all attributes, and ‘doc_string’ SHOULD be used on all attributes. An attribute MUST have only one of the value-carrying properties.

#### Values

The representation distinguishes between two kinds of values: attribute values, which are statically known, and runtime values. The type of values permitted in the two cases are different. The permitted types of attribute values are indicated by the enumeration `AttributeType`, while the permitted types of runtime values are described by `TypeProto`. 

The types of the inputs and outputs of the model must be specified, including the shapes of tensors. While the ranks of input and output tensors are statically specified, the sizes of specific dimensions (axis) may be statically unknown and are indicated so using symbolic identifiers in the shape. 

#### Optional Inputs

Some operators have inputs that are marked as optional. There are two ways to leave an optional input unspecified. The first is to simply not provide that input. However, this is not always possible - for example, if you wish to leave the fourth input unspecified, but still provide a value for the fifth input. Therefore, any input with a name of the empty string is treated as an unspecified optional input.

## Standard data types

There are two official ONNX variants; the main distinction between the two is found in the supported types. The neural-network-only __ONNX__ definition recognizes only tensors as input and output types, while the Classical Machine Learning extension, __ONNX-ML__ also recognizes sequences and maps.

The following data types are supported by ONNX for inputs and outputs of graphs and nodes. They also define the types used for the initializer values of a graph.

Primitive numeric, string, and Boolean types MUST be used as elements of tensors. Maps and sequences MUST contain tensors as values.

### Tensor Element Types

| | | 
|---|---|---|
Floating Point Types|float16, float32, float64|Values adhering to the IEEE 754-2008 standard representation of floating-point data.
Signed Integer Types|int8, int16, int32, int64|Signed integers are supported for 8-64 bit widths.
Unsigned Integer Types|uint8, uint16|Unsigned integers of 8 or 16 bits are supported.
Complex Types|complex64, complex128|A complex number with either 32- or 64-bit real and imaginary parts.
Other|string|Strings represent textual data. All strings are encoded using UTF-8.
Other|bool|Boolean value represent data with only two values, typically true and false.

### Input / Output Data Types

The following types are used to define the types of graph and node inputs and outputs.

|Variant | Type | Description | 
|---|---|---|
ONNX|dense tensors|Tensors are a generalization of vectors and matrices; whereas vectors have one dimension, and matrices two, tensors can have any number of dimensions, including zero. A zero-dimensional tensor is logically equivalent to a scalar value.
ONNX-ML|sequence|Sequences represent dense, ordered, collections of elements that are of homogeneous types.
ONNX-ML|map|Maps represent associative tables, defined by a key type and a value type.


#### Tensor shapes

In addition to element type and dense/sparse properties, tensors have shape. A shape is a list of sizes that define whether the tensor is a vector, a matrix, or a higher-dimensioned structure. For example, a 100x100 matrix would have the shape [100,100].

The empty list of sizes, [], is a valid tensor shape. It's denotes a scalar value. A zero-dimension tensor is distinct from a tensor of unknown dimensionality.

Each size in the list MUST be expressed as an integral value or as a "dimension variable," a string denoting that the actual size of the dimension is not statically constrained to a particular number, which is useful for declaring interfaces that care about the number of dimensions, but not the exact size of each dimension. 

For example, a NxM matrix would have the shape list [N,M].

The name of each dimension variable MUST adhere to C identifier syntax.

Dimension variables are scoped to the declaration (graph signature, node signature, or single operator declaration) that they appear in. Thus, any given name denotes the same value within a declaration, allowing a declaration to describe how the shapes of inputs and outputs are related. For example, a graph that performs matrix cross-product may be defined as taking two inputs of shape [K,M] and [M,N], and produce an output of shape [K,N].

Shapes MAY be defined using a combination of integers and variables.

### Attribute Types

The type system used for attributes is a superset of that used for of inputs and outputs. In addition to tensors, attribute values may be scalar numerical values, strings, and graphs. Sequences are available for attributes in both ONNX and ONNX-ML. Maps are not available for attributes in either variant. 

## Other Specification Documents 

The ONNX and ONNX-ML specification is comprised of this document, which defines the semantics of the IR and the standard data types, and the following documents defining the operator semantics and the IR syntax. The syntax is specified as Protobuf v2 and v3 schema files.

### Operators

[Neural Network Operators](Operators.md)

[Classical Machine Learning operators](Operators-ml.md)

### Syntax

[ONNX Models and Graphs - protobuf v2](../onnx/onnx.proto)

[ONNX Models and Graphs - protobuf v3](../onnx/onnx.proto3)

[ONNX-ML Models and Graphs - protobuf v2](../onnx/onnx-ml.proto)

[ONNX-ML Models and Graphs - protobuf v3](../onnx/onnx-ml.proto3)

[ONNX Operator Sets - protobuf v2](../onnx/onnx-operators.proto)

[ONNX Operator Sets - protobuf v3](../onnx/onnx-operators.proto3)

[ONNX-ML Operator Sets - protobuf v2](../onnx/onnx-operators-ml.proto)

[ONNX-ML Operator Sets - protobuf v3](../onnx/onnx-operators-ml.proto3)
