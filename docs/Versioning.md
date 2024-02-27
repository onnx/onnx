<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Versioning

This document describes the rules for versioning ONNX. MUST, SHOULD et al are used consistent with [RFC2119](https://tools.ietf.org/html/rfc2119).

## Versioning Principles

ONNX defines the versioning policy and mechanism for three classes of entities:

* The [intermediate representation (IR) specification](IR.md), which is the abstract model for graphs and operators and the concrete format that represents them. These are always versioned atomically and are referred to as the *IR version*.
* Operator specifications that may be referenced by a given ONNX graph. We refer to this as the *operator version*.
* A defined/trained model that defines a specific graph in terms of specific operators. We refer to this as the *model version*.

The versioning of all three of these entity types is distinct and largely independent. The IR specification evolves at a different (generally slower) rate than the operator specifications. Model versions are entirely independent of the other two versions.

Specific policies for version management are mandated only for IR version and operator version. For model versioning, they are merely recommendations. For model versioning, ONNX users and systems MAY follow whichever local customs make sense; however, to facilitate easily managing shared collections of ONNX models, they SHOULD adhere to the policies described under model versioning.

New IR and operator versions are released as part of ONNX _releases_, which have their own versioning scheme. The release versioning scheme is not described as part of the standard itself. It is discussed in the [ONNX release management document](../RELEASE-MANAGEMENT.md).

### Semantic Versioning or Simple Numbers?

The ONNX versioning system allows for simple monotonically increasing numbers or [semantic versioning (SemVer)](https://semver.org/). For IR and operator sets, versioning is based on simple numbers. For models, ONNX does not require any scheme, but recommends a set of shared conventions.

Which versioning scheme is in use by a model is made clear by inspecting the most significant four bytes, which MUST be non-zero when using semantic versioning and MUST be zero when using simple numbers. In other words, when using SemVer, at least one of the MAJOR or MINOR numbers must be non-zero.

### SemVer, Files and Consumers

For model and release versioning, ONNX builds on the principles and syntax defined by [SemVer 2.0.0](http://semver.org/spec/v2.0.0.html). Throughout this document, we use the terms *breaking change*, *non-breaking change*, and *patch* consistent with SemVer 2.0.0.

Because ONNX models are serialized files (not APIs), it's worth making clear the relationship between a serialized model and a piece of software that consumes that model. As a rough approximation, the serialized model plays the role of an API's *callee*, while the consumer of the serialized model plays the role of the API's *caller*.

The ONNX versioning principles are based on the [robustness principle](https://en.wikipedia.org/wiki/Robustness_principle): "be conservative in what you do, be liberal in what you accept from others".

1. A producer of a given ONNX model (and the ONNX specification itself) MUST strictly adhere to the rules for breaking vs. non-breaking changes defined in this specification.
2. A consumer of a given ONNX model SHOULD consume an updated ONNX file, provided there are no breaking changes in the new ONNX file's IR version, referenced operator versions, or model version (meaning the MAJOR version numbers have not changed between the two ONNX files).
3. A consumer of a given ONNX model MAY consume an updated ONNX file, provided there are one or more breaking changes in the new ONNX file's IR version, referenced operator versions, or model version.

### Serializing SemVer version numbers in protobuf

For efficiency, ONNX serializes the MAJOR, MINOR, and PATCH values as a bit-packed 64-bit integer; the two most significant bytes are the MAJOR component, the next two most significant bytes are the MINOR component, and the least significant four bytes are the PATCH component.

For example, *1.2.345* is represented as *0x0001000200000159*.

Pre-release and build metadata are not stored in the model.

## IR versioning

The IR format is versioned using simple numbers, which MUST be monotonically increasing. Breaking changes to the format or semantics of the ONNX specification require an increment of the version. Non-breaking changes to the IR format do not require changing the version number.

NOTE: breaking changes include those that do not alter the serialized binary format, but still break software using libraries that write or read it. For example, changing the spelling of a message property will cause code accessing the property break.

The IR format adheres to the versioning guidelines defined in the [Updating a Message Type](https://developers.google.com/protocol-buffers/docs/proto3#updating) section of the proto3 specification.

As a general principle, implementations SHOULD be robust in the face of missing fields. However, to ensure basic interoperation, a subset of message fields will be marked as required for a given IR version and all producers MUST set these fields correctly. Required fields MUST always be marked with the comment:

    // This field MUST be present for this version of the IR.

For example, the `ModelProto.ir_version` property MUST be present in every model. The ONNX checker (`onnx/checker.py`) will enforce these rules.

Because the protocol buffer message definitions (.proto / .proto3 files) are expected to be consumed by multiple independent developers, changes to those definitions SHOULD NOT break code that depends on generated language bindings (e.g., changing the type of an existing field).

## Operator versioning

The IR can evolve independently from the set of operators. Operators represent both the signature and semantics of a given operation. Operators are abstract interfaces in that they do not imply a specific implementation; rather, they are simply the contract between a model author and the implementations that model may execute on.

A given operator is identified by a three-tuple: `(domain, op_type, since_version)`, written as `domain.op_type:since_version` in prose (e.g., `com.acme.FastConv:3`). `since_version` is the version of the operator set that introduced the operator. Breaking operator changes include:

* Adding/removing/renaming an attribute. This even includes the case of adding a new optional attribute, where omitting the attribute would imply a default value yielding semantics identical to the previous operator version.

* Adding/removing/reordering inputs or outputs.

* Adding/removing types supported by inputs and outputs, and changing types used by attributes.

* Supporting new behavior even when the existing parameter signature is otherwise identical (e.g. implicitly supporting tensor broadcasting in the Mean operator).

The following are not breaking:

* Clarifications of specification ambiguities to match prevailing
  implementation practice.

Changes to the semantics of an operator or function MUST be introduced in a new operator, which MUST be introduced in a new [operator set](#operator-sets).

> In practice, this means that BC-breaking changes in the ONNX
> repository require contributors to follow these steps:
>
> 1. Increment the maximum version in `DomainToVersionRange`.
> 2. Copy the old operator schema to an `old.cc` file.
> 3. Update the `SinceVersion` signifier to the new max version from
>    step (1).
> 4. Register the new operator in the corresponding `operator_sets`
>    header file.
> 5. Add a version adapter to `convert.h` so that the version
>    converter can upgrade the old version of the operator to the new
>    one. This can be a `CompatibleAdapter` in case operators following
>    the old schema are still valid under the new one (which is usually
>    true).
> 6. A version adapter to downgrade the new operator to the older version
>    can also be added to `convert.h` but it's not mandatory.

How nodes bind to operator declarations is strictly defined, and are designed to increase model compatibility across ONNX implementations, in the spirit of the conservative clause of the robustness principle.

How ONNX implementations bind an operator declaration to a specific implementation is outside the scope of this specification. Implementations of ONNX MAY elect to introduce more sophisticated operator declaration/implementation binding modes, in the spirit of the liberal clause of the robustness principle.

### Operator sets

ONNX uses operator sets to group together immutable operator specifications. An operator set represents a specific version of a domain, indicated by a pair (domain, version). This represents the set of all operators belonging to the specified domain with the specified version (referred to as the `opset_version`). When the inventory of a given operator set changes either by addition, removal, or a change in semantics of a contained operator, its version MUST increase.

Models declare which operator sets they require as a list of `(domain, opset_version)` pairs in `ModelProto.opset_import`. The empty string ("") domain indicates the operators defined as part of the ONNX specification; other domains correspond to operator sets of other vendors (meaning they can be used to provide vendor-specific extensions to ONNX). The union of the operator sets specified by a given model MUST have a compatible operator declaration for each node in the model's graph.

### Example

This section is not normative and informational only.

Given the following operator sets:

OpSet|Operators|Comments
-|-|-
1|{A}           | A introduced
2|{A, B}        | B introduced
3|{A', B, C}    | A updated (to A'), C introduced
4|{B, C'}       | A removed, C updated (to C')

The operators for a given operator set will have the following `since_version` values:

Operator|OpSet 1|OpSet 2|OpSet 3|OpSet 4
-|-|-|-|-
A|**1** |1      |**3**  |**-**
B|-     |**2**  |2      |2
C|-     |-      |**3**  |**4**

Notes:
- Values that are new or updated from a previous OpSet version are in **bold**.

## Model versioning

This section of the specification is not normative. It simply outlines a set of recommended practices.

Model authors and applications/systems MAY elect to ignore the model versioning mechanism and policy rules. For models that will be shared across developers, teams, or organizations, model authors and applications/systems SHOULD adhere to the following version policies:

### Signature Changes

1. Breaking changes to the ModelProto.graph.GraphProto.input or .output MUST increment the MAJOR version of `ModelProto.model_version`. Breaking changes include:

    * Breaking changes to the semantics of an input or output (e.g., changing the required contents of an input tensor from a color image to a black and white image).
    * Changing the declared type of an input or output to an incompatible type (e.g., `tensor(int)->tensor(string)`).
    * Adding a new input for which there is no meaningful or specified default value. Recall that default values for inputs are specified in the initializer list.
    * Removing an existing output for which there is no meaningful or specified default value.

2. Non-breaking changes to the ModelProto.graph.GraphProto.input or .output MUST increment the MINOR version of `ModelProto.model_version`. Non-breaking changes include:

    * Changing the declared type of an input or output to a compatible/widening type (e.g., `tensor(int32)->tensor(int64)`, `tensor(float16)->tensor(float32)`).
    * Adding a new input for which there is a meaningful or specified default value.
    * Adding new behavior that is only triggered in the presence of inputs that were not
    possible in prior versions of the graph (typically by the presence of a new input
    or allowing a previously invalid input value).

### Accuracy or performance changes

Changes that impact accuracy or performance significantly but do not change the model's inputs or outputs SHOULD increment the PATCH version of `ModelProto.model_version`.

## Released Versions

ONNX version|IR version|Opset version ai.onnx|Opset version ai.onnx.ml|Opset version ai.onnx.training
------------|-------------------|---------------------|------------------------|------------------------------
1.0|3|1|1|-
1.1|3|5|1|-
1.1.2|3|6|1|-
1.2|3|7|1|-
1.3|3|8|1|-
1.4.1|4|9|1|-
1.5.0|5|10|1|-
1.6.0|6|11|2|-
1.7.0|7|12|2|1
1.8.0|7|13|2|1
1.8.1|7|13|2|1
1.9.0|7|14|2|1
1.10.0|8|15|2|1
1.10.1|8|15|2|1
1.10.2|8|15|2|1
1.11.0|8|16|3|1
1.12.0|8|17|3|1
1.13.0|8|18|3|1
1.13.1|8|18|3|1
1.14.0|9|19|3|1
1.14.1|9|19|3|1
1.15.0|9|20|4|1
1.16.0|10|21|5|1

A programmatically accessible version of the above table is available [here](../onnx/helper.py). Limited version number
information is also maintained in [version.h](../onnx/common/version.h) and [schema.h](../onnx/defs/schema.h).
Please update all whenever a new version of ONNX is released.
