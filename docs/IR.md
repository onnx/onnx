<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Open Neural Network Exchange Intermediate Representation (ONNX IR) Specification

__Purpose__

This document contains the normative specification of the semantics of ONNX.

The `.proto` and `.proto3` files found under the [onnx folder](/onnx/) form the normative specification of its syntax authored in the [Protocol Buffers](https://developers.google.com/protocol-buffers) definition language. Commentary found in the `.proto` and `.proto3` files are intended to improve readability of those files, but are not normative if they conflict with this document. Such conflicts should be reported as documentation bugs.

__Notes on model validation__

A [tool](../onnx/checker.py) is available to perform general validation of models against this specification. It is implemented in C++ with a Python command-line wrapper.

__Notes on language in this and all related documents__:

1. The use of SHOULD, MUST, MAY and so on in this document is consistent with [RFC 2119](https://www.ietf.org/rfc/rfc2119.txt).

2. The use of 'list' shall denote an ordered collection of items, 'set' shall denote an unordered collection of unique elements, and 'bag' an unordered collection of possibly non-unique elements.

## Components

ONNX is an open specification that consists of the following components:

1)  A definition of an extensible computation graph model.

2)  Definitions of standard data types.

3)  Definitions of built-in operators.

#1 and #2 together make up the ONNX Intermediate Representation, or 'IR', specification which is covered herein; the built-in operators are covered in documents listed at the end. Specifically, built-in operators are divided into a set of primitive operators and functions. A function is an operator whose semantics is formally expressed via expansion into a sub-graph (called the function body) using other operators (and functions). Functionality-wise, an ONNX compatible framework or runtime may inline a function body to execute it if it does not have corresponding implementation of the function.

There are two official ONNX variants; the main distinction between the two is found in the default operator sets. __ONNX-ML__ extends the __ONNX__ operator set with ML algorithms that are not based on neural networks.

Up to IR version 6, the ONNX specification and model format addressed only inference (also known as scoring). Starting from IR version 7, the ONNX specification and model format also support training. An ONNX training model is an extension of the inference-model. An inference-only runtime can consume a training model ignoring the training-related extensions. However, an inference-only model may enable a representation that is more optimal for inference purposes than a training model.

## Runtime Agnostic

ONNX does not pre-suppose or imply any particular method of runtime implementation.

For example, an implementation may consist of a rich runtime which interprets the model; it may be a code generator that translates the model in its entirety to executable code for some target programming language; it may be a hardware implementation; it may be a combination of two or three of those.

Nothing in this specification should be construed as advocating one implementation approach over any other; any comments on the inner workings of concrete implementations are to be interpreted as examples.

## ONNX Versioning

The IR specification, individual models, and operator sets are all versioned. Furthermore, each individual operator indicates which version of its containing operator set it was introduced or stabilized in.

Version numbers can be used as a simple number, or used to encode [semantic versions](https://semver.org/)(AKA SemVer). If using semantic versions, the convention is to use the two most significant bytes for the major number, the next two bytes for the minor number, and the least significant four bytes for the patch/build/bugfix number. When using semantic versioning, at least one of the major/minor numbers MUST be non-zero.

The IR specification uses simple monotonically increasing numbers for its versions. The valid IR versions are defined by the `onnx.Version` enumeration in [onnx.proto](/onnx/onnx.proto).

Operator sets use a simple version number. Each operator set version represents a snapshot of the set of operators, and their semantics at a particular point in time.

This specification does not provide guidance on what versioning scheme model producers should be using.

More details on conventions and best practices for versioning of IR, operator sets, and models can be found in [Versioning](Versioning.md).

## Extensible computation graph model

ONNX specifies the portable, serialized format of a computation graph. It does not have to be the form a framework chooses to use internally. For example, an implementation may represent the model differently in memory if it is more efficient to manipulate during optimization passes.

An implementation MAY extend ONNX by adding operators expressing semantics beyond the standard set of operators that all implementations MUST support. The mechanism for this is adding operator sets to the `opset_import` property in a model that depends on the extension operators.

### Models

The top-level ONNX construct is a ‘Model.’, and is represented in protocol buffers as the type `onnx.ModelProto`

The main purpose of the model structure is to associate metadata with a graph which contains all the executable elements. The metadata is used when first reading the model file, giving an implementation the information it needs in order to determine whether it will be able to execute the model, generate logging messages, error reports, etc. Further, the metadata is useful to tools, such as IDEs and model galleries, which need it for informing humans about a given model’s purpose and characteristics.

Each model has the following components:

|Name|Type|Description|
|---|---|---|
|ir_version|int64|The ONNX version assumed by the model.|
|opset_import|OperatorSetId|A collection of operator set identifiers made available to the model. An implementation must support all operators in the set or reject the model.|
|producer_name|string|The name of the tool used to generate the model.|
|producer_version|string|The version of the generating tool.|
|domain|string|A reverse-DNS name to indicate the model namespace or domain, for example, 'org.onnx'|
|model_version|int64|The version of the model itself, encoded in an integer.|
|doc_string|string|Human-readable documentation for this model. Markdown is allowed.|
|graph|Graph|The parameterized graph that is evaluated to execute the model.|
|metadata_props|map<string,string>|Named metadata values; keys should be distinct.|
|training_info|TrainingInfoProto[]|An optional extension that contains information for training.|
|functions|FunctionProto[]|An optional list of functions local to the model.|

 Models MUST specify a domain and use reverse domain names based on the responsible organization's identity, the same convention that is used for [naming Java packages](https://docs.oracle.com/javase/tutorial/java/package/namingpkgs.html).

__Note: Exploring an ONNX file__

You can use the `protoc` tool that is part of the Protocol Buffers distribution to examine the contents of an ONNX file, you do so like this:

```
$ protoc --decode=onnx.ModelProto onnx.proto < yourfile.onnx
```

Where [onnx.proto](/onnx/onnx.proto) is the file that is part of this repository.

Alternatively, you can use a tool like [Netron](https://github.com/lutzroeder/netron) to explore the ONNX file.

### Model Semantics

The semantics of an inference-model is a _stateless function_ (except possibly for the state used for random-number generation). Thus, whenever an inference-model (without random-generator operations) is used to perform inference on the same input, it is expected to produce the same output.

The semantics of a training model is that of a _stateful object_, with the state consisting of the current values of trained-weights (and any other auxiliary state required, such as momentum, for example, used by the learning algorithm). Specifically, its semantics is captured via three methods: an initialization method (which is used to initialize or reset the values of state variables), a training step method (to train using a batch of input-output pairs), and an inference method to perform inference using the current values of the learned weights. The first two methods update the state of the object, while the third method is side-effect-free.

### Optional Metadata

The 'metadata_props' field in the model is available for any kind of optional metadata that a tool or model developer chooses to place there. The following are the defined “standard” optional metadata properties of a model.

Name|Type|Format|Description
|---|---|---|---|
model_author|string|A comma-separated list of names.|The personal name of the author(s) of the model, and/or their organizations.
model_license|string|Name or URL.|The well-known name or URL of the license under which the model is made available.

### Operator Set Identifiers

Each operator set is uniquely identified by a (domain, version) pair.

Name|Type|Description
|---|---|---|
domain|string|The domain of the operator set being identified.
version|int64|The version of the operator set being identified. Same as 'opset_version' in the operator set.

The operator set version is a simple integer value that is monotonically increased as new versions of the operator set are published.

Operator sets other than the default operator set MUST specify a domain and SHOULD use reverse domain names based on the responsible organization's identity, the same convention that is used for [naming Java packages](https://docs.oracle.com/javase/tutorial/java/package/namingpkgs.html).

### Operator Sets

Each model MUST explicitly name the operator sets that it relies on for its functionality. Operator sets define the available operators and their version. Each model defines the imported operator sets by their domains. All models implicitly import the default ONNX operator set.

Each operator set SHALL be defined in a separate document, also using protobuf as the serialization format. How operator set documents are found at runtime is implementation-dependent.

__Note: As of the publication of this document, no ONNX implementation is known to process operator set documents.__

The properties of an operator set are:

Name|Type|Description
|---|---|---|
magic|string|The value ‘ONNXOPSET’
ir_version|int32|The ONNX version corresponding to the operators.
ir_version_prerelease|string|The prerelease component of the SemVer of the IR.
ir_build_metadata|string|The build metadata of this version of the operator set.
domain|string|The domain of the operator set. Must be unique among all sets.
opset_version|int64|The version of the operator set.
doc_string|string|Human-readable documentation for this operator set. Markdown is allowed.
operator|Operator[]|The operators contained in this operator set.

The operator set version is a simple integer value that is monotonically increased as new versions of the operator set are published.

Operator sets other than the default operator set MUST specify a domain and SHOULD use reverse domain names based on the responsible organization's identity, the same convention that is used for [naming Java packages](https://docs.oracle.com/javase/tutorial/java/package/namingpkgs.html).

### Operators

Each operator used within a graph MUST be explicitly declared by one of the operator sets imported by the model.

The properties of an operator definition are:

Name|Type|Description
|---|---|---|
op_type|string|The name of the operator (case sensitive), as used in graph nodes. MUST be unique within the operator set’s domain.
since_version|int64|The version of the operator set when this operator was introduced.
status|OperatorStatus|One of ‘EXPERIMENTAL’ or ‘STABLE.’
doc_string|string|A human-readable documentation string for this operator. Markdown is allowed.

The version value MUST be the same value as the operator set version when the operator was first published. Subsequent versions of the operator set MUST NOT alter the signature or semantics of the operator once published as STABLE.

The ‘status’ property indicates whether the syntax, semantics, or presence of the operator is in an experimental or stable stage. Once an operator is published as STABLE, it’s syntax and semantics MUST NOT change in subsequent versions of the operator set.

There are two distinct ways to pass information to operators – inputs and attributes. Inputs represent graph inputs or values computed elsewhere in the graph, while attributes are used for values that are constants in the graph. This distinction may be highly relevant to achieving good performance for some implementations, while completely irrelevant to others.

### Functions

A _function_ may be thought of as an operator combined with an implementation of the operator using
other, more primitive, ops, referred to as the _function body_. The function body consists of a
topologically sorted list of nodes that form a graph. Thus, a function combines aspects of both
an operator as well a graph (described below).

Each function contained in a Model (also referred to as a model-local function) serves
as a default or fallback implementation of the corresponding operator. A runtime, however,
may choose to use an alternative implementation of the operator (usually as an optimized kernel).
As such, the unique name of a function is significant as it is implicitly associated with a
semantic specification.

A serialized function (a _FunctionProto_) has the following properties:

|Name|Type|Description|
|---|---|---|
name|string|The name of the function
domain|string|The domain to which this function belongs
overload|string|Part of unique id of function (added in IR version 10)
doc_string|string|Human-readable documentation for this function. Markdown is allowed.
attribute|string[]|The attribute parameters of the function
attribute_proto|Attribute[]| (IR version 9+) The attribute parameters with default values of the function. A function attribute shall be represented either as a string attribute or an Attribute, not both.
input|string[]|The input parameters of the function
output|string[]|The output parameters of the function.
node|Node[]|A list of nodes, forming a partially ordered computation graph. It must be in topological order.
|opset_import|OperatorSetId|A collection of operator set identifiers used by the function implementation.
|value_info|ValueInfo[]| (IR version >= 10) Used to store the type and shape information of values used in the function.
|metadata_props|map<string,string>|(IR version >= 10) Named metadata values; keys should be distinct.

The name and domain serve to identify the operator uniquely in IR versions upto 9. IR version 10 adds the
field overload, and the triple (name, domain, overload) acts as a unique-id across functions stored in
a model. This is intended to support cases where distinct function-bodies are required for distinct
calls to the function within the model.
An opset version is not explicitly identified in a FunctionProto, but it is implicitly determined by
the opset version of the domain included in the model.

The input, output, attribute, and attribute_proto (added in IR version 9) constitute the signature part of the operator. No type information
is explicitly included in the signature. The attribute_proto field describes attribute parameters of the function along with their default-value (when not specified by an call-site node), while the attribute field lists attribute parameters without a default-value. The names in these two lists must be distinct. When an attribute-parameter of the function is used in a node within the function, it is replaced by the actual parameter value specified for the attribute at a call-site node (of the function) when such a attribute is specified, and it is replaced by the default-value if the attribute has a default-value specified, and it is omitted otherwise.

The opset_import and node fields describe the implementation of the function.

The value_info field (added in IR version 10) allows a model to store type and shape information about the values used in a function, including its inputs and outputs. Note that this is optional, and ONNX allows functions to be polymorphic.

### Graphs

A graph is used to describe a side-effect-free computation (function).
A serialized graph is comprised of a set of metadata fields, a list of model parameters, and a list of computation nodes.

Each computation dataflow graph is structured as a topologically sorted list of nodes that form a graph, which MUST be free of cycles. Each node represents a call to an operator or a model local function. Each node has zero or more inputs and one or more outputs.

Graphs have the following properties:

|Name|Type|Description|
|---|---|---|
name|string|The name of the model graph.
node|Node[]|A list of nodes, forming a partially ordered computation graph based on input/output data dependencies. It is in topological order.
initializer|Tensor[]|A list of named tensor values. When an initializer has the same name as a graph input, it specifies a default value for that input. When an initializer has a name different from all graph inputs, it specifies a constant value. The order of the list is unspecified.
doc_string|string|Human-readable documentation for this model. Markdown is allowed.
input|ValueInfo[]|The input parameters of the graph, possibly initialized by a default value found in ‘initializer.’
output|ValueInfo[]|The output parameters of the graph. Once all output parameters have been written to by a graph execution, the execution is complete.
value_info|ValueInfo[]|Used to store the type and shape information of values that are not inputs or outputs.
|metadata_props|map<string,string>|(IR version >= 10) Named metadata values; keys should be distinct.

ValueInfo has the following properties:

Name|Type|Description
|---|---|---|
name|string|The name of the value/parameter.
type|Type|The type of the value **including shape information**.
doc_string|string|Human-readable documentation for this value. Markdown is allowed.

Each main (top-level) graph MUST define the names, types and shapes of its inputs and outputs, which are specified as ‘value info’ structures. The main graph inputs and outputs are required to have a shape, indicating the rank, even though the exact dimensions need not be specified.

Nested subgraphs (specified as attribute values) MUST define the names of its inputs and outputs
and MAY define the types of its inputs and outputs.

Each graph MUST specify a name.

The graph MUST adhere to single static assignment (SSA) for all node outputs; this means that all node output names MUST be unique within a graph.

Graphs SHOULD be populated with documentation strings, which MAY be interpreted using GitHub-style markdown syntax. HTML and other text-markup languages MAY NOT be used in documentation strings.

### Names Within a Graph

All names MUST adhere to [C90 identifier syntax rules](https://en.cppreference.com/w/c/language/identifier).

Names of nodes, inputs, outputs, initializers, and attributes are organized into several namespaces. Within a namespace, each name MUST be unique for each given graph. Please see below for further clarification in the case where a graph contains nested subgraphs (as attribute values).

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
input|string[]|Names of the values used by the node to propagate input values to the node operator. It must refer to either a graph input, a graph initializer or a node output.
output|string[]|Names of the outputs used by the node to capture data from the operator invoked by the node. It either introduces a  value in the graph or refers to a graph output.
op_type|string|The symbolic identifier of the operator to invoke.
domain|string|The domain of the operator set that contains the operator named by the op_type.
attribute|Attribute[]|Named attributes, another form of operator parameterization, used for constant values rather than propagated values.
doc_string|string|Human-readable documentation for this value. Markdown is allowed.
overload|string|Part of unique id of function (added in IR version 10)
|metadata_props|map<string,string>|(IR version >= 10) Named metadata values; keys should be distinct.

A name belonging to the Value namespace may appear in multiple places, namely as a graph input, a graph initializer, a graph output, a node input, or a node output. The occurrence of a name as a graph input, a graph initializer, or as a node output is said to be a definition and the occurrence of a name as a node input or as a graph output is said to be a use.

A value name used in a graph must have a unique definition, with the exception that the same name MAY appear in both the graph input list and graph initializer list. (Further exceptions apply in the presence of nested subgraphs, as described later.)

When a name appears in both the initializer list and the graph input list, a runtime MAY allow a caller to specify a value for this (input) name overriding the value specified in the initializer and a runtime MAY allow users to omit specifying a value for this (input) name, choosing the value specified in the initializer. Names of constants that are not meant to be overridden by the caller should appear only in the initializer list and not in the graph input list. In models with IR version >= 4, in nested subgraphs used as attribute values, users MUST NOT use the same name as both a subgraph initializer and subgraph input unless the corresponding op's specification explicitly allows it. In models with IR version <= 3, users MAY use the same name as both a subgraph initializer and subgraph input, but this is restricted to support constants via initializers that are not intended to correspond to any actual inputs passed from the node into the subgraph. In particular, the control-flow operator semantics determines the set of inputs supplied to the execution of the subgraph, and these input names MUST NOT appear as subgraph initializers. Subgraph initializer names must appear in the graph input list _after_ the actual inputs. This allows the actual inputs and formal inputs to be matched positionally.

Edges in the computation graph are established by outputs of one node being referenced by name in the inputs of a subsequent node.

The outputs of a given node introduce new names into the graph. The values of node outputs are computed by the node's operator. Node inputs MAY refer to node outputs, graph inputs, and graph initializers. When the name of a node output coincides with the name of a graph output, the graph output's value is the corresponding output value computed by that node. A node input in a nested subgraph MAY refer to names introduced in outer graphs (as node outputs, graph inputs, or graph initializers).

The graph MUST use single static assignment for all node outputs, which means that all node output names MUST be unique within a graph. In the case of a nested subgraph, a node output name MUST be distinct from the names from the outer scopes that are visible in the nested subgraph.

Node dependencies MUST NOT create cycles in the computation graph.

The number of inputs and outputs in a node, their types, the set of attributes specified in a node and their types MUST satisfy the constraints imposed by the signature of the node’s operator.

The list of nodes defining the top-level computation graph MUST be ordered topologically; that is, if node K follows node N in the graph, none of the data inputs of N may refer to outputs of K.

Node attributes are used to pass literal (static) values to operators.

#### Input and Output Values

The representation distinguishes between two kinds of values: attribute values, which are statically known, and input/output values. The types of values permitted in the two cases are different.

Input and output values are found as graph inputs, outputs, and initializers, and as node inputs and outputs. Their values are determined at runtime, either by the code that initiates model execution, or by operators computing output values.

#### Attributes

Attribute values are only found in nodes, passed to operators by name association. Attribute values are runtime constants, in that their values are determined when a model graph is constructed and therefore not computed at runtime. A common use for attributes is to represent coefficients established during model training.

Attributes have the following properties:

Name|Type|Description
|---|---|---|
name|string|The name of the attribute. Must be unique among attributes, inputs, and outputs for any given operator and node.
doc_string|string|Human-readable documentation for this value. Markdown is allowed.
type|AttributeType|The type of the attribute, determining which of the remaining fields is used to hold the value of the attribute.
f|float|A 32-bit floating-point value.
i|int64|A 64-bit integer value.
s|byte[]|UTF-8 string.
t|Tensor|A tensor value.
g|Graph|A graph.
floats|float[]|A list of 32-bit floating-point values.
ints|int64[]|A list of 64-bit integer values.
strings|byte[][]|A list of UTF-8 strings.
tensors|Tensor[]|A list of tensor values.
graphs|Graph[]|A list of graphs.
ref_attr_name|string|The name of a parent function's attribute.

The properties ‘name’ and ‘type’ are required on all attributes, and ‘doc_string’ SHOULD be used on all attributes. An attribute MUST have only one of the value-carrying properties.

In case ‘ref_attr_name’ is set, this attribute does not contain data, and instead it's a reference to the parent function's attribute of the given name. Can only be used within the function body.


#### Variadic Inputs and Outputs

The last input or output of an operator MAY be marked as variadic. For example, the operator 'Max()' can be used to compute the maximum of a varying number of input values. A variadic operator has a minimum arity, which specifies the minimum number of operands that must be specified.

For each variadic operator input, N or more node inputs must be specified where N is the minimum arity of the operator. For each variadic operator output, N or more node outputs must be specified where N is the minimum arity of the operator.

#### Optional Inputs and Outputs

**Pre IR-8**

Some operators have inputs that are marked as optional, which means that a referring node MAY forgo providing values for such inputs.

Some operators have outputs that are optional. When an actual output parameter of an operator is not specified, the operator implementation MAY forgo computing values for such outputs.

There are two ways to leave an optional input or output unspecified: the first, available only for trailing inputs and outputs, is to simply not provide that input; the second method is to use an empty string in place of an input or output name.

Each node referring to an operator with optional outputs MUST provide a name for each output that is computed and MUST NOT provide names for outputs that are not computed.

Optional inputs and outputs of the above kind are referred to as _static-optional_.

**IR-8 Version and Later**

IR-8 introduced a new type-constructor to represent _dynamic-optional_ inputs and outputs,
in addition to the earlier static-optional version described above. A dynamic-optional INT64
tensor is a distinct type from an INT64 tensor type. In contrast, a static-optional INT64
tensor does not have a distinct type, it has the same type as a INT64 tensor.
The ops _Optional_ and _OptionalGetElement_ must be explicitly used to convert between
the dynamic-optional type and the underlying non-optional type.
The dynamic-optional allows for more expressiveness than static-optional.

#### External Tensor Data

The raw data for large constant tensors, such as initializers, MAY be serialised in a separate file. In such a case, the tensor MUST provide the filename relative to the model file and MUST NOT use the value fields. It MAY provide a byte offset and length within that file. It MAY also specify a SHA1 digest of the file. One file MAY contain the data for multiple tensors.

More details can be found in [External Data](ExternalData.md).

## Standard data types

There are two official ONNX variants; the main distinction between the two is found in the supported types and the supported operators.

With respect to supported types, both __ONNX__ and __ONNX-ML__ definition recognize tensors, sparse tensors, sequences, maps, and optionals as input and output types. Sequences and maps were supported from the IR version 6 (ONNX 1.6.0 release). Optional type was supported from IR vesion 8 (ONNX 1.10.0 release).

The following data types are supported by ONNX for inputs and outputs of graphs and nodes as well as the initializers of a graph.

Primitive numeric, string, and Boolean types MUST be used as elements of tensors.

### Tensor Definition

Tensors are a generalization of vectors and matrices; whereas vectors have one dimension, and matrices two, tensors can have any number of dimensions, including zero. A zero-dimensional tensor is logically equivalent to a scalar value.

Mathematically, a tensor can be defined as a pair of sequences/lists (V, S) where S is the shape of the tensor (a list of non-negative integers) and V is a list of values with length equal to the product of the dimensions in S. Two tensors (V, S) and (V', S') are equal if and only if V = V' and S = S'. The length of S is referred to as the rank.

 - If S has length 0, V must have length 1, since the empty product is defined to be 1. In this case, the tensor represents a scalar.
 - S can contain dimensions of value 0. If any dimensions are 0, V must have length 0.
 - If S has length 1, V has length equal to the single dimension in S. In this case, the tensor represents a vector.
 - A tensor representing a vector of length 1 has shape [1], while a tensor representing a scalar has shape []. They both have a single element, but scalars are _not_ vectors of length 1.

A tensor's shape S is a list but can be represented as a tensor with values S and shape [R] where R is the rank of the tensor.

 - For a tensor (V, S), the tensor representing its shape is (S, [R]).
 - The shape of a scalar is []. Represented as a tensor, [] has shape [0].

#### Representation

It is common to represent a tensor as a nested list. This generally works fine, but is problematic when zero dimensions are involved. A tensor of shape (5, 0) can be represented as [[], [], [], [], []], but (0, 5) is represented as [] which loses the information that the second dimension is 5.

 - A nested list is not a complete representation of a tensor with dimensions of value zero.

### Tensor Element Types

|Group|Types|Description|
|---|---|---|
Floating Point Types|float16, float32, float64, bfloat16, float8e4m3fn, float8e5m2, float8e4m3fnuz, float8e5m2fnuz|Values adhering to the IEEE 754-2008 standard representation of floating-point data or defined in papers [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) and [8-bit Numerical Formats for Deep Neural Networks](https://arxiv.org/abs/2206.02915)
Signed Integer Types|int4, int8, int16, int32, int64|Signed integers are supported for 4-64 bit widths.
Unsigned Integer Types|uint4, uint8, uint16, uint32, uint64|Unsigned integers are supported for 4-64 bit widths.
Complex Types|complex64, complex128|A complex number with either 32- or 64-bit real and imaginary parts.
Other|string|Strings represent textual data. All strings are encoded using UTF-8.
Other|bool|Boolean values represent data with only two values, typically true and false.

### Input / Output Data Types

The following types are used to define the types of graph and node inputs and outputs.

|Variant | Type | Description |
|---|---|---|
ONNX|dense tensors|Represents a Tensor. See definition above.
ONNX|sequence|Sequences are dense, ordered, collections of elements that are of homogeneous types.
ONNX|map|Maps are associative tables, defined by a key type and a value type.
ONNX|optional|Optionals are wrappers that may contain an element of tensor, sequence, or map type, or may be empty (containing none). [Details](ONNXTypes.md)

#### Static tensor shapes

In addition to element type, tensor types have a **static** shape. The static shape of a tensor variable is related to, but different from, the runtime (dynamic) shape of a tensor value. A static tensor shape is a list of records that indicates whether the tensor is a vector, a matrix, or a higher-dimensional value. For example, a 100x100 matrix has the shape [100,100].

The static shape is defined by 'TensorShapeProto':

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

The empty list of dimension sizes, [], is a valid tensor shape, denoting a zero-dimension (scalar) value. A zero-dimension tensor is distinct from a tensor of unknown dimensionality, which is indicated by an absent 'shape' property in the Tensor message. When the shape property is absent in the type of a value (including node input),
it indicates that the corresponding runtime value may have any shape. This sub-section describes how to interpret a missing-shape or a shape with missing dimensions etc. However, specific usage contexts may impose further constraints on a type and shape.
For example, the inputs and outputs of a model (top-level graph) are required to *have* a shape, indicating the rank of inputs and outputs,
even though the exact dimensions need not be specified.

Each size in the list MAY be expressed as an integral value or as a "dimension variable," a string denoting that the actual size of the dimension is not statically constrained to a particular number. This is useful for declaring interfaces that care about the number of dimensions, but not the exact size of each dimension. A dimension MAY have neither dim_value nor dim_param set. Such a dimension represents an unknown dimension unrelated to other unknown dimensions.

For example, a NxM matrix would have the shape list [N,M].

The name of each dimension variable MUST adhere to [C90 identifier syntax rules](https://en.cppreference.com/w/c/language/identifier).

Currently, dimension variables are not scoped. A dimension variable "N" represents the same value across the entire graph in a model. For example, if the graph has two inputs X and Y each with shape ["N"], then at runtime the values passed in for X and Y MUST be tensors of rank 1 with the same dimension. Nested sub-graphs currently share the same scope for dimension variables as the main-graph. This allows a model to relate the dimensions of tensors inside the subgraph to the dimensions of tensors in the outer graph.

ONNX supports types such as Sequences of Tensors. The global scoping of dimension variables means that a variable with type "Sequence<Tensor<float, [M,N]>" represents a sequence of tensors that *all have the same shape*. The dimension variables M or N must be omitted from the above type if that dimension does not have a fixed size across all tensors in the sequence. The entire shape must be omitted from the type if different tensors in the sequence may have different ranks.

For example, a graph that performs matrix cross-product may be defined as taking two inputs of shape [K,M] and [M,N], and producing an output of shape [K,N].

Shapes MAY be defined using a combination of integers and variables.

_Historical Notes_: The following extensions were considered early on, but were never implemented or supported.
* The use of an empty string (as a dimension variable) to denote an unknown dimension not related to any other dimension. This was discarded in favor of using a Dimension with neither dim_value nor dim_param set.
* The use of the string "\*" (as a dimension variable) to denote a sequence of zero or more dimensions of unknown cardinality. This is not supported. In the current implementation, the number of dimensions in a shape MUST represent the rank of the tensor. A tensor of unknown rank is represented using a TypeProto::Tensor object with no shape, which is legal.
* A scoping mechanism to allow dimension variables that are local to a sub-graph (such as the body of a loop) may be useful, but is not currently supported.
* ONNX supports types such as Sequences of Tensors. A scoping mechanism for the dimension variables local to a type may be useful to distinguish between the following two types: a sequence of square matrices (of differing sizes) vs a sequence of square matrices (all of same size). This is not currently supported.

### Attribute Types

The type system used for attributes is related to but slightly different from that used for of inputs and outputs. Attribute values may be a dense tensor, sparse tensor, a scalar numerical value, a string, a graph, or repeated values of one of the above mentioned types.

## Other Metadata

The ModelProto structure, and in IR versions >= 10, various other structures (GraphProto, FunctionProto, NodeProto)
contain a metadata_props field allowing users to store other metadata in the form of key-value pairs.
It is recommended that users use key names qualified with a reverse-DNS name as prefix
(such as "ai.onnxruntime.key1") to avoid conflicts between different uses.
Unqualified names may be used in the future by the ONNX standard.

## Training Related Information

Training related information is described by one or more instances of _TrainingInfoProto_ contained in a model. Each TrainingInfoProto contains information describing both an initialization step and a training step.

The initialization step is described using a Graph (TrainingInfoProto.initialization) and an initialization-binding map (TrainingInfoProto.initialization_binding). The initialization step is performed by evaluating the Graph, and assigning the outputs produced by the Graph to the _state variables_ of the training model as specified in the initialization-binding. The initialization-binding is conceptually a map, specified as a list of key-value pairs, where each key is the name of a state variable, and the value is the name of an output of the (initialization) Graph. Each name specified as a key in the binding MUST be the name of an initializer that appears in the main inference graph (i.e., in ModelProto.graph.initializer) or the name of an initializer that appears in TrainingInfoProto.algorithm.initializer. Each name specified as a value in the binding MUST be the name of an output of the TrainingInfoProto.initialization graph. Key values specified in the repeated initialization_binding field MUST be unique.

The training step is similarly described using a Graph (TrainingInfoProto.algorithm) and an update-binding map (TrainingInfoProto.update_binding). The training step is performed by evaluating the Graph and assigning the outputs produced by the Graph to the state variables as specified in the update-binding. The constraints and description presented above for the initialization apply to the training step as well.

Thus, the state variables of the training model consist of a subset of the initializers of the main inference graph (i.e., ModelProto.graph.initializer) and the training-algorithm graph (TrainingInfoProto.algorithm.initializer) as identified by the keys of the bindings (in TrainingInfoProto.initialization_binding and TrainingInfoProto.update_binding). Note that the state variables are not constant values in the context of training. They represent mutable variables shared by multiple graphs (implicitly declared in the top-level training model scope). This implicit declaration of shared mutable variables is used instead of an explicit declaration for purposes of backward compatibility with the inference graph representation.

All state variables are pre-initialized to the value specified in the corresponding initializer. A subsequent call to perform the initialization step (using the appropriate API exposed by a runtime) updates the values of the state variables as described above. If the training model has more than one instance of TrainingInfoProto, the initialization step corresponding to each is performed in order. A TrainingInfoProto.initialization MAY be omitted (only if there are no initialization_bindings). For the training step, a runtime MAY allow users to invoke any one of the TrainingInfoProto.algorithm, allowing the training process to interleave the different algorithms as desired. The order in which the different TrainingProto.algorithms are called affects the training result, and it is the callers responsibility to call them in the correct order.

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
