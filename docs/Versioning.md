# ONNX versioning

This document describes the rules for versioning ONNX. Like the rest of the ONNX
specification, MUST, SHOULD et al are used consistent with [RFC2119](https://tools.ietf.org/html/rfc2119).

## Versioning Principles

ONNX defines versioning policy and mechanism for three classes of entities:

* The abstract model for graphs and operators and the concrete format that represents them. These are always versioned atomically and are referred to as the *IR version.* 
* Operator specifications that may be referenced by a given ONNX graph. We refer to this as the *operator version*.
* An defined/trained model that defines a specific graph in terms of specific operators. We refer to this version as the *model version.* 

The versioning of all three of these entity types is distinct, and largely independent. That is,  the ONNX IR format evolves at a different rate than the set operators defined by ONNX - the former will version much slower than the latter.

While the versioning mechanisms are clearly specified in this document, the policies for version management are only mandated for IR version and operator verison.

For model version, ONNX users and systems MAY follow whatever local customs make sense, however, to facilitate shared libraries or repositories of ONNX models to be managable, models SHOULD adhere to the policies described under Model versioning

### SemVer, Files and Frameworks

ONNX builds on the principles and syntax defined by [SemVer 2.0.0](http://semver.org/spec/v2.0.0.html). Throughout this document, we use the terms *breaking change*, *non-breaking change*, and *patch* consistent with SemVer 2.0.0.

The SemVer specification is written in terms of the dependencies between the caller and callee of a given API.  With respect to operator definitions, the NodeProto represents the caller, and the operator specification represents the callee.

Because ONNX models are serialized files, not APIs, it's worth making clear how the dependency between a serialized model and a piece of software that consumes that model relate.  As a rough approximation, the serialized model plays the role of an API's *callee*; the consumer of the seriaized model plays the role of the API's *caller.*

The ONNX versioning principles are based on [Postel's law](https://en.wikipedia.org/wiki/Robustness_principle): be conservative in what you do, be liberal in what you accept from others.

1. A producer of a given ONNX model (and the ONNX specification itself) MUST strictly adhere to the rules for breaking vs. non-breaking changes defined in this specification.
2. A consumer of a given ONNX model SHOULD consume an updated ONNX file provided there are no breaking changes in the new ONNX file's IR version, referenced operator versions, or model version (e.g., the MAJOR version numbers are have not changed between the two ONNX files) .
3. A consumer of a given ONNX model MAY consume an updated ONNX file provided there are one or more breaking changes in the new ONNX file's IR version, referenced operator versions, or model version.

The operational rules for how the ONNX project is managed are documented at [here](../RELEASE-MANAGEMENT.md).

### Serializing SemVer version numbers in protobuf
For efficiency, ONNX serializes the MAJOR, MINOR, and PATCH values as a bit-packed 32-bit integer; the most siginificant byte is the MAJOR component, the second most significant byte is the MINOR component, the least significant two bytes are the PATCH component.

For example, 1.2.345 is represented as 0x01020159.

The prerelease and build metadata aren't stored in the model.

## IR versioning

Changes to the file format or abstract graph semantics version atomically. Breaking changes to the format or semantics of the ONNX specification require an increment of the MAJOR version.  Non-breaking format or semantic changes that introduce new functionality require an increment of the MINOR version. Non-breaking changes to the specification that simply clarify spec ambiguities require an increment of the PATCH version.

The ONNX IR format adheres to the versioning guidelines defined in the [Updating a Message Type](https://developers.google.com/protocol-buffers/docs/proto3#updating) section of the proto3 specification.

As a general principle, implementations SHOULD be robust in the face of missing fields. However, to ensure basic interoperation, a subset of message fields will be marked as required for a given IR version and all producers MUST set these fields correctly. Required fields always are marked with the comment:

    // This field MUST be present for this version of the IR.

By way of example, the `ModelProto.ir_version` MUST be present in every model.  The ONNX checker (`onnx/checker.py`) will enforce these rules.

Because onnx.proto is expected to be consumed by multiple independent developers, changes to onnx.oroto SHOULD NOT break code that depends on generated language bindings (e.g., changing the type of an existing field).

ISSUE: define type compatiblity rules either here or under model versioning - probably here

## Operator versioning

ONNX is defined such that the IR can evolve independently from the set of operators. In ONNX, operators represent both the signature and semantics of a given operation.  Operators are abstract interfaces in that they do not imply a specific implementation; rather, they are simply the contract between a model author and the implementations that model may execute on. 

A given operator is identified by a three-tuple: (domain, op_type, op_version), written domain.op_type:op_version in prose (e.g., com.acme.FastConv:3).  Nodes in graphs always refer
to operators by their three-part identifier.

Once an operator is published, all implementations of that operator's (domain, op_type, op_version) MUST adhere to the signature and semantics of the operator at the time of publication. 
Any change of semantics implies a new operator, which MAY share a domain and op_type with another operator. This includes adding new behavior triggered only by previously unspecified inputs or attributes - these changes in semantics also MUST have a distinct operator id.

ONNX uses operator sets to group together immutable operator specifications. An ONNX operator set
specifies both the domain of all operators it includes, as well as an opset version. The opset version is largely independent from the version field of the operators it includes. When the enventory of a given operator set changes either by addition or removal, its opset version MUST increase. Moreover,
the opset version MUST be no less than the highest operator version number in the set.

ONNX models declare which operator sets they require as a list of two part operator ids (domain, opset_version).  The empty string ("") domain indicates the operators defined as part of the 
ONNX specification. The union of the operator sets specified by a given model MUST have have have a compatible operator declaration for each node in the model's graph.  


How nodes bind to operator declarations is strictly defined and are designed to increase model compatibility across ONNX implementations (appealing to the conservative clause of the robustnes principle). 

How ONNX implementations bind an operator declaration to specific implementation is outside the scope of this specification.
Implementations of ONNX MAY elect to introduce more sophisticated operator declaration/implementation binding modes to appeal to the liberal clause of the robustness principle.

## Model versioning

Model versioning is ultimately the domain of a given organization, and therefor, this section of the spec is non-normative and simply proposes a set of practices to consider.

Model authors and applications/systems  MAY elect to ignore model versioning mechanism and policy rules. For models that will be shared across developers, teams, or organizations, model authors and applications/systems SHOULD adhere to the following version policies:

ISSUE: the following is a strawman. I'm confident some of it is right and some is wrong. Either way, we need to make some calls and document it.  Also note that A LOT of this will likely apply to operators, as both operators and graphs have signatures AND can be versioned.

### Signature Changes
1. Breaking changes to the ModelProto.graph.GraphProto.input or .output MUST increment the MAJOR version of `ModelProto.model_version`. Breaking changes include:

    * Breaking changes to the semantics of an input or output (e.g., changing the required contents of an input tensor from color image to black and white image).
    * Changing the declared type of an input or output to an incompatible type (e.g., `tensor(int)->tensor(string)`).
    * Adding a new input for which there is no meaningful or specified default value. For graph inputs, those values are provided by a same-named value in GraphProto. initializer.
    * Removing an existing output for which there is no meaningful or specified default value.

2. Non-breaking changes to the ModelProto.graph.GraphProto.input or .output MUST increment the MINOR version of `ModelProto.model_version`. Non-breaking changes include:

    * Changing the declared type of an input or output to an compatible/widening type (e.g., `tensor(int32)->tensor(int64)`, `tensor(float16)->tensor(float32)`.
    * Adding a new input for which there is a meaningful or specified default value.
    * Adding new behavior that is only triggered in the presence of inputs that were not
    possible in prior versions of the graph (typically by the presense of a new input
    or allowing a previously invalid input value).

### IR version/Operator version dependency changes

ISSUE: what's our policy when a model takes a dependency on new `IR_VERSION` change and/or new operator change?


### Accuracy or performance changes

Assuming that there are no breaking changes to the signature of the model's graph or any operator dependencies, the shape and contents of the graph can change freely provided there are no semantic changes to the model. However, changes to the shape and contents of the graph can impact model accuracy and/or model performance.

