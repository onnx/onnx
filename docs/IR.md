Open Neural Network Exchange - ONNX
=========

__Purpose__

This document contains the normative specification of the semantics of ONNX. The .proto and .proto3 files found under the ‘onnx’ folder form the normative specification of its syntax. Commentary found in the .proto and .proto3 files are intended to improve readability of those files, but are not normative if they conflict with this document. Such conflicts should be reported as documentation bugs.

__Notes on model validation__

A [tool](../onnx/checker.py) is available to perform general validation of models against this specification. It is implemented in C++ with Python command-line wrapper.

__Notes on language in this and all related documents__:

1. The use of SHOULD, MUST, MAY and so on in this document is consistent with [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

2. The use of 'list' shall denote an ordered collection of items, 'set' shall denote an unordered collection of unique elements, and 'bag' an unordered collection of possibly non-unique elements.

## Components

ONNX is an open specification that consists of the following components:

1)  A definition of an extensible computation graph model.

2)  Definitions of standard data types.

3)  Definitions of built-in operators.

Of these, #1 and #2 are covered herein; the built-in operators are covered separately in documents listed at the end of this.

There are two official ONNX variants; the main distinction between the two is found in the supported types and the default operator sets. The neural-network-only __ONNX__ variant recognizes only tensors as input and output types, while the Classical Machine Learning extension, __ONNX-ML__, also recognizes sequences and maps. __ONNX-ML__ extends the __ONNX__ operator set with ML algorithms that are not based on neural networks.

## Runtime Agnostic

ONNX does not pre-suppose or imply any particular method of runtime implementation. 

For example, an implementation may consist of a rich runtime which interprets the model; it may be a code generator that translates the model in its entirety to executable code for some target programming language; it may be a hardware implementation; it may be a combination of two or three of those. 

Nothing in this specification should be construed as advocating one implementation approach over any other; any comments on the inner workings of concrete implementations are to be interpreted as examples.

## ONNX Versioning

Versioning features in several places in ONNX -- the IR specification itself, the version of a model, and the version of an operator set. Futhermore, each individual operator indicates which version of its containing operator set it was introduced or stabilized in.

Version numbers can be used as a simple number, or used to encode semantic versions. If using semver, the convention is to use the two most significant bytes for the major number, the next two bytes for the minor number, and the least significant four bytes for the build/bugfix number. When using semver versioning, at least one of the major/minor numbers MUST be non-zero.

The IR specification uses simple monotonically increasing numbers for its versions. The valid IR versions is defined by an enumeration, which currently has the following values:
```
  //  Version 1, published on Oct 10, 2017.
  IR_VERSION_2017_10_10 = 0x0000000000000001;

  // Version 2, published on Oct 30, 2017
  IR_VERSION_2017_10_30 = 0x0000000000000002;

  // Version 3 published on Nov 3, 2017
  IR_VERSION = 0x0000000000000003;
```

Operator sets use a simple version number. Each operator set version represents the combination of the most recent version of each operator.

This specification does not provide guidance on what versioning scheme model producers should be using.

More details on conventions and best practices for versioning of IR, operator sets, and models can be found in [Versioning](Versioning.md).

## Extensible computation graph model

ONNX specifies the portable, serialized format of a computation graph. It does not have to be the form a framework chooses to use and manipulate the computation internally. For example, an implementation may represent the model differently in memory if it is more efficient to manipulate during optimization passes.

An implementation MAY extend ONNX by adding operators expressing semantics beyond the standard set of operators that all implementations MUST support. The mechanism for this is adding operator sets to the opset_import property in a model that depends on the extension operators.

### Models

The top-level ONNX construct is a ‘Model.’

The main purpose of the model structure is to associate metadata with a graph, which is what contains all the executable elements. The metadata is used when first reading the model file, giving an implementation the information that it needs in order to determine whether it will be able to execute the model, generate logging messages, error reports, etc. Further, the metadata is useful to tools, such as IDEs and model galleries, which need it for informing humans about a given model’s purpose and characteristics.

Each model has the following components:

|Name|Type|Description|
|---|---|---|
|ir_version|int64|The ONNX version assumed by the model.|
|opset_import|OperatorSetId|A collection of operator set identifiers made available to the model. An implementation must support all operators in the set or reject the model.|
|producer_name|string|The name of the tool used to generate the model.|
|producer_version|string|A string representing the version of the generating tool.|
|domain|string|A reverse-DNS name to indicate the model namespace or domain, for example, 'org.onnx'|
|model_version|int64|A version of the model itself, encoded in an integer.|
|doc_string|string|A human-readable documentation for this model. Markdown is allowed.|
|graph|Graph|The parameterized graph that is evaluated to execute the model.|
|metadata_props|map<string,string>|Named metadata values; keys should be distinct.|

 Models MUST specify a domain and use reverse domain names based on the responsible organization's identity, the same convention that is traditionally used for naming Java packages.
 

### Optional Metadata

The 'metadata_props' field in the model is available for any kind of optional metadata that a tool or model developer chooses to place there. The following are the defined “standard” optional metadata properties of a model. 

Name|Type|Format|Description
|---|---|---|---|
model_author|string|A comma-separated list of names.|The personal name of the author(s) of the model, and/or their organizations.
model_license|string|Name or URL.|The well-known name or URL of the license under which the model is made available.


### Operator Sets

Each model MUST explicitly name the operator sets that it relies on for its functionality. Operator sets define the available operators, their version, and their status. Each model defines the imported operator sets by their domains. All models implicitly import the default ONNX operator set.

Each operator set SHALL be defined in a separate document, also using protobuf as the serialization format. How operator set documents are found at runtime is implementation-dependent.

__Note: As of the publication of this document, no ONNX implementation is known to process operator set documents.__

The properties of an operator set are:

Name|Type|Description
|---|---|---|
magic|string|The value ‘ONNXOPSET’
ir_version|int32|The ONNX version corresponding to the operators.
ir_version_prerelease|string|The prerelease component of the SemVer of the IR.
ir_build_metadata|string|The symbolic identifier of the operator to invoke.
domain|string|The domain of the operator set. Must be unique among all sets.
opset_version|int64|The version of the set of operators. 
doc_string|string|A human-readable documentation for this set of operators. Markdown is allowed.
operator|Operator[]|The operators of this operator set.

The operator set version is a simple integer value that is monotonically increased as new versions of the operator set are published.

Operator sets other than the default operator set MUST specify its domain and SHOULD use reverse domain names based on the responsible organization's identity, the same convention that is used for [naming Java packages](https://docs.oracle.com/javase/tutorial/java/package/namingpkgs.html).

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

There are two distinct ways to pass information to operators – inputs and attributes. The latter are used for values that are constants in the graph, while the former represent graph inputs or values computed elsewhere in the graph. This distinction may be highly relevant to achieving good performance for some implementations, while completely irrelevant to others.

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
value_info|ValueInfo|Used to store the type and shape information of values that are not inputs or outputs.

Each graph MUST define the names and types of its inputs and outputs, which are specified as ‘value info’ structures, having the following properties:

Name|Type|Description
|---|---|---|
name|string|The name of the value/parameter.
type|Type|The type of the value.
doc_string|string|A human-readable documentation for this value. Markdown is allowed.

Each graph MUST specify a name.

The graph MUST adhere to single static assignment (SSA) for all node outputs; this means that all node output names MUST be unique within a graph.

Graphs SHOULD be populated with documentation strings, which MAY be interpreted using GitHub-style markdown syntax. HTML and other text-markup languages MAY NOT be used in documentation strings.

### Names Within a Graph

All names MUST adhere to C identifier syntax rules.

Names of nodes, inputs, outputs, initializers, and attributes are organized into several namespaces. Within a namespace, each name MUST be unique for each given graph. 

The namespaces are:

Namespace|Description
|---|---|
Attribute|The names of attributes of an operator. Unique for each operator.
Value|The names of values – node inputs & outputs, tensor values (if named), graph inputs, outputs.
Node|The names of graph nodes.
Graph|The names of graphs within a domain, unique within the model domain.
Operator|The names of operators within a domain.
Shape|The names of tensor shape variables – scoped to the value information records of a graph, which is where shape variables occur.


### Nodes

Computation nodes are comprised of a name, the name of an operator that it invokes, a list of named inputs, a list of named outputs, and a list of attributes. 

Input and outputs are positionally associated with operator inputs and outputs. Attributes are associated with operator attributes by name.

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

The outputs of a given node introduce new names into the graph. The values of node outputs are computed by the node's operator. Node inputs MAY refer to node outputs, graph inputs, and graph initializers. When the name of a node output coincides with the name of a graph output, the graph output's value is the corresponding output value computed by that node.

The graph MUST use single static assignment for all node outputs, which means that all node output names MUST be unique within a graph.

Node dependencies MUST NOT create cycles in the computation graph.

The number of inputs and outputs in a node, their types, the set of attributes specified in a node and their types MUST satisfy the constraints imposed by the signature of the node’s operator.

The list of nodes defining the top-level computation graph MUST be ordered topologically; that is, if node K follows node N in the graph, none of the data inputs of N may refer to outputs of K.

Node attributes are used to pass literal (static) values to operators.

#### Input and Output Values

The representation distinguishes between two kinds of values: attribute values, which are statically known, and input/output values. The types of values permitted in the two cases are different.

Input and output values are found as graph inputs, outputs, and initializers, and as node inputs and outputs. Their values are determined at runtime, either by the code that initiates model execution, or by operators computing output values.

#### Attributes

Attribute values are only found in nodes, passed to operators by name association. Attribute values are runtime constants, in that their values are determined when a model graph is constructed and therfore not computed at runtime. A common use for attributes is to represent coefficients established during model training.

Attributes have the following properties:

Name|Type|Description
|---|---|---|
name|string|The name of the attribute. Must be unique among attributes, inputs, and outputs for any given operator and node.
doc_string|string|A human-readable documentation for this value. Markdown is allowed.
type|AttributeType|The type of the attribute, determining which of the remaining fields is used to hold the value of the attribute.
f|float|A 32-bit floating-point value.
i|int64|A 64-bit integer value.
S|byte[]|UTF-8 string.
t|Tensor|A tensor value.
g|Graph|A graph.
floats|float[]|A list of 32-bit floating-point values.
ints|int64[]|A list of 64-bit integer values.
strings|byte[][]|A list of UTF-8 strings.
tensors|Tensor[]|A list of tensor values.
graphs|Graph[]|A list of graphs.

The properties ‘name’ and ‘type’ are required on all attributes, and ‘doc_string’ SHOULD be used on all attributes. An attribute MUST have only one of the value-carrying properties.


#### Variadic Inputs and Outputs
 
The last input or output of an operator MAY be marked as variadic. For example, the operator 'Max()' can be used to compute the maximum of a varying number of input values.

For each variadic operator input, one or more node inputs must be specified. For each variadic operator output, one or more node outputs must be specified. 

#### Optional Inputs and Outputs

Some operators have inputs that are marked as optional, which means that a referring node MAY forgo providing values for such inputs.

Some operators have outputs that are optional. When an actual output parameter of an operator is not specified, the operator implementation MAY forgo computing values for such outputs. 

There are two ways to leave an optional input or output unspecified: the first, available only for trailing inputs and outputs, is to simply not provide that input; the second method is to use an empty string in place of an input or output name.

Each node referring to an operator with optional outputs MUST provide a name for each output that is computed and MUST NOT provide names for outputs that are not computed.


## Standard data types

There are two official ONNX variants; the main distinction between the two is found in the supported types and the supported operators.

With respect to supported types, the __ONNX__ definition recognizes only tensors as input and output types, while the Classical Machine Learning extension. __ONNX-ML__, also recognizes sequences and maps.

The following data types are supported by ONNX for inputs and outputs of graphs and nodes as well as the the initializers of a graph.

Primitive numeric, string, and Boolean types MUST be used as elements of tensors.

### Tensor Element Types

|Group|Types|Description| 
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

ONNX currently does not define a sparse tensor type.

#### Tensor shapes

In addition to element type, tensors have shape. A tensor shape is a list of records that define whether the tensor is a vector, a matrix, or a higher-dimensional value. For example, a 100x100 matrix has the shape [100,100].

The shape record is defined by 'TensorShapeProto':

```
message TensorShapeProto {
  message Dimension {
    oneof value {
      int64 dim_value = 1;
      string dim_param = 2;
    };
  };
  repeated Dimension dim = 1;
}
```
Which is referenced by the Tensor type message:

```
  message Tensor {
    optional TensorProto.DataType elem_type = 1;
    optional TensorShapeProto shape = 2;
  }
```

The empty list of dimension sizes, [], is a valid tensor shape, denoting a zero-dimension (scalar) value. A zero-dimension tensor is distinct from a tensor of unknown dimensionality, which is indicated by an absent 'shape' property in the Tensor record. When the shape property is absent for an input, a tensor value of any shape may be passed from the caller. When the shape property is absent for an output, the caller should expect that the output value may be of any shape.

Each size in the list MUST be expressed as an integral value or as a "dimension variable," a string denoting that the actual size of the dimension is not statically constrained to a particular number. This is useful for declaring interfaces that care about the number of dimensions, but not the exact size of each dimension. 

For example, a NxM matrix would have the shape list [N,M].

The name of each dimension variable MUST adhere to C identifier syntax.

Dimension variables are scoped to the declaration that they appear in. For graph inputs and outputs, the graph itself is the declaration. Consequently, any name that is repeated denotes the same value within a declaration, allowing a declaration to describe how the shapes of inputs and outputs are related. Dimension variables appearing in a graph's 'value_info' record are scoped to each value, allowing each value to have its shape defined independently. 

For example, a graph that performs matrix cross-product may be defined as taking two inputs of shape [K,M] and [M,N], and producing an output of shape [K,N].

The emptry string "", when used as a dimension name, denotes a single dimension of any cardinality. The string "*", when used as a dimension name, denotes zero or more dimensions of unknown cardinality.

Shapes MAY be defined using a combination of integers and variables.

### Attribute Types

The type system used for attributes is a superset of that used for of inputs and outputs. In addition to tensors, attribute values may be scalar numerical values, strings, and graphs. Sequences are available for attributes in both ONNX and ONNX-ML. Maps are not available for attributes in either variant. 

## Other Specification Documents 

The ONNX specification is comprised of this document, which defines the semantics of the IR and the standard data types, and the following documents defining standard operator semantics and the IR syntax. The latter is specified as Protobuf v2 and v3 schema files.

See the [metadata category documentation](MetadataProps.md) for more details.

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

### Versioning Conventions and Best Practices

[Versioning](Versioning.md)
