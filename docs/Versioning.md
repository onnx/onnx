# ONNX versioning

This document describes the rules for versioning ONNX. Like the rest of the ONNX
specification, MUST, SHOULD et al are used consistent with [RFC2119](https://tools.ietf.org/html/rfc2119).

## Versioning Principles

ONNX defines the versioning policy and mechanism for three classes of entities:

* The abstract model for graphs and operators and the concrete format that represents them. These are always versioned atomically and are referred to as the *IR version*. 
* Operator specifications that may be referenced by a given ONNX graph. We refer to this as the *operator version*.
* A defined/trained model that defines a specific graph in terms of specific operators. We refer to this version as the *model version*. 

The versioning of all three of these entity types is distinct and largely independent. That is, the ONNX IR format evolves at a different rate than the set operators defined by ONNX – in which the former will version much slower than the latter.

While the versioning mechanisms are clearly specified in this document, specific policies for version management are mandated only for IR version and operator version. For model versioning, they are merely recommendations. For model version, ONNX users and systems MAY follow whichever local customs make sense; however, to facilitate easily managing shared collections of ONNX models, they SHOULD adhere to the policies described under model versioning.

In addition to versioning ONNX entities, progressive ONNX _releases_ are assigned increasing version numbers. The release versioning scheme is not described as part of the standard itself. It is discussed in the [ONNX release management document](../RELEASE-MANAGEMENT.md).

### Semantic Versioning or Simple Numbers?

The ONNX versioning system allows for simple monotonically increasing numbers or semantic versioning. For IR and operator sets, versioning is based on simple numbers. For models, ONNX does not proscribe one or the other methodology, but (as stated earlier) recommends a set of shared conventions.

Which versioning scheme is in use by a model is made clear by inspecting the most significant four bytes, which MUST be non-zero when using semantic versioning and MUST be zero when using simple numbers. In other words, when using semver, at least one of the MAJOR or MINOR numbers must be non-zero.

### SemVer, Files and Frameworks

For model and release versioning, ONNX builds on the principles and syntax defined by [SemVer 2.0.0](http://semver.org/spec/v2.0.0.html). Throughout this document, we use the terms *breaking change*, *non-breaking change*, and *patch* consistent with SemVer 2.0.0.

Because ONNX models are serialized files (not APIs), it's worth making clear how the dependency between a serialized model and a piece of software that consumes that model relate.  As a rough approximation, the serialized model plays the role of an API's *callee*, while the consumer of the serialized model plays the role of the API's *caller*.

The ONNX versioning principles are based on [Postel's law](https://en.wikipedia.org/wiki/Robustness_principle) – be conservative in what you do, be liberal in what you accept from others.

1. A producer of a given ONNX model (and the ONNX specification itself) MUST strictly adhere to the rules for breaking vs. non-breaking changes defined in this specification.
2. A consumer of a given ONNX model SHOULD consume an updated ONNX file, provided there are no breaking changes in the new ONNX file's IR version, referenced operator versions, or model version (e.g., the MAJOR version numbers have not changed between the two ONNX files).
3. A consumer of a given ONNX model MAY consume an updated ONNX file, provided there are one or more breaking changes in the new ONNX file's IR version, referenced operator versions, or model version.

The operational rules specifying how the ONNX project is managed are documented [here](../RELEASE-MANAGEMENT.md).

### Serializing SemVer version numbers in protobuf

For efficiency, ONNX serializes the MAJOR, MINOR, and PATCH values as a bit-packed 64-bit integer; the two most significant byte is the MAJOR component, the second two most significant byte is the MINOR component, the least significant four bytes are the PATCH component.

For example, *1.2.345* is represented as *0x0001000200000159*.

Pre-release and build metadata are not stored in the model.

## IR versioning

The IR file format is versioned using simple numbers, which MUST be monotonically increasing. Breaking changes to the format or semantics of the ONNX specification require an increment of the version. Non-breaking changes to the IR format do not require changing the version number. 

NOTE: breaking changes include those that do not alter the serialized binary format, but still break software using libraries that write or read it. For example, changing the spelling of a message property will cause code accessing the property break.

The ONNX IR format adheres to the versioning guidelines defined in the [Updating a Message Type](https://developers.google.com/protocol-buffers/docs/proto3#updating) section of the proto3 specification.

As a general principle, implementations SHOULD be robust in the face of missing fields. However, to ensure basic interoperation, a subset of message fields will be marked as required for a given IR version and all producers MUST set these fields correctly. Required fields MUST always be marked with the comment:

    // This field MUST be present for this version of the IR.

For example, the `ModelProto.ir_version` property MUST be present in every model. The ONNX checker (`onnx/checker.py`) will enforce these rules.

Because onnx.proto is expected to be consumed by multiple independent developers, changes to onnx.oroto SHOULD NOT break code that depends on generated language bindings (e.g., changing the type of an existing field).

ISSUE: define type compatibility rules either here or under model versioning - probably here

## Operator versioning

ONNX is defined such that the IR can evolve independently from the set of operators. In ONNX, operators represent both the signature and semantics of a given operation.  Operators are abstract interfaces in that they do not imply a specific implementation; rather, they are simply the contract between a model author and the implementations that model may execute on.

A given operator is identified by a three-tuple: `(domain, op_type, and op_version)`. This is written as `domain.op_type:op_version` in prose (e.g., `com.acme.FastConv:3`).  Nodes in graphs always refer to operators by their three-part identifier. Breaking opset changes include:

* Adding/removing/renaming an attribute. This even includes the case of adding a new optional attribute, where omitting the attribute would imply a default value yielding semantics identical to the previous operator version.

* Adding/removing/reordering inputs or outputs.

* Adding/removing types supported by inputs and outputs, and changing types used by attributes.

* Supporting new behavior even when the existing parameter signature is otherwise identical (e.g. implicitly supporting tensor broadcasting in the Mean operator).

The following are not breaking:

* Clarifications of specification ambiguities to match prevailing
  implementation practice.

If the semantics of an operator or function are changed, you MUST create a new operator; the `op_version` of the new
operator id MUST be greater than any extant `op_version` for the
`domain`.

> In practice, this means that BC-breaking changes in the ONNX
> repository require contributors to follow the following three steps:
>
> 1. Increment the maximum version in `DomainToVersionRange`,
> 2. Copy the old operator schema to an `old.cc` file, and
> 3. Update the `SinceVersion` signifier to the new max version from
>    step (1).

ONNX uses operator sets to group together immutable operator specifications. An ONNX operator set specifies both the domain of all operators it includes, as well as an opset version. The opset version is largely independent from the version field of the operators it includes. When the inventory of a given operator set changes either by addition or removal, its opset version MUST increase. Moreover, the opset version MUST be no less than the highest operator version number in the set.

ONNX models declare which operator sets they require as a list of two part operator ids (domain, opset_version).  The empty string ("") domain indicates the operators defined as part of the  ONNX specification; other domains correspond to operator sets of other vendors (e.g., they can be used to provide vendor-specific extensions to ONNX). The union of the operator sets specified by a given model MUST have a compatible operator declaration for each node in the model's graph.

How nodes bind to operator declarations is strictly defined, and are designed to increase model compatibility across ONNX implementations (appealing to the conservative clause of the robustness principle). 

How ONNX implementations bind an operator declaration to specific implementation is outside the scope of this specification. Implementations of ONNX MAY elect to introduce more sophisticated operator declaration/implementation binding modes to appeal to the liberal clause of the robustness principle.

## Model versioning

Model versioning is ultimately the domain of a given organization. Therefore, this section of the specification is not normative. It simply outlines a set of recommended practices.

Model authors and applications/systems MAY elect to ignore the model versioning mechanism and policy rules. For models that will be shared across developers, teams, or organizations, model authors and applications/systems SHOULD adhere to the following version policies:

ISSUE: the following is a strawman. I'm confident some of it is right and some is wrong. Either way, we need to make some calls and document it.  Also note that A LOT of this will likely apply to operators, as both operators and graphs have signatures AND can be versioned.

### Signature Changes

1. Breaking changes to the ModelProto.graph.GraphProto.input or .output MUST increment the MAJOR version of `ModelProto.model_version`. Breaking changes include:

    * Breaking changes to the semantics of an input or output (e.g., changing the required contents of an input tensor from a color image to a black and white image).
    * Changing the declared type of an input or output to an incompatible type (e.g., `tensor(int)->tensor(string)`).
    * Adding a new input for which there is no meaningful or specified default value. For graph inputs, those values are provided by a same-named value in GraphProto. initializer.
    * Removing an existing output for which there is no meaningful or specified default value.

2. Non-breaking changes to the ModelProto.graph.GraphProto.input or .output MUST increment the MINOR version of `ModelProto.model_version`. Non-breaking changes include:

    * Changing the declared type of an input or output to an compatible/widening type (e.g., `tensor(int32)->tensor(int64)`, `tensor(float16)->tensor(float32)`.
    * Adding a new input for which there is a meaningful or specified default value.
    * Adding new behavior that is only triggered in the presence of inputs that were not
    possible in prior versions of the graph (typically by the presence of a new input
    or allowing a previously invalid input value).

### IR version/Operator version dependency changes

ISSUE: what's our policy when a model takes a dependency on new `IR_VERSION` change and/or new operator change?


### Accuracy or performance changes

Assuming that there are no breaking changes to the signature of the model's graph or any operator dependencies, the shape and contents of the graph can change freely provided there are no semantic changes to the model. However, changes to the shape and contents of the graph can impact model accuracy and/or model performance.

## Released Versions

ONNX version|File format version|Operator set version ai.onnx|Operator set version ai.onnx.ml
------------|-------------------|----------------------------|-------------------------------
1.0|3|1|1
1.1|3|5|1
1.1.2|3|6|1
1.2|3|7|1
1.3|3|8|1
