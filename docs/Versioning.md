# ONNX versioning

This document describes the rules for versioning ONNX. Like the rest of the ONNX
specification, MUST, SHOULD et al are used consistent with [RFC2119](https://tools.ietf.org/html/rfc2119).

## Versioning Principles

ONNX defines versioning policy and mechanism for three classes of entities:

* The abstract graph model and the concrete format that serializes it. The graph model and format are versioned atomically and are referred to as the *IR version.* The IR version is represented by the `ModelProto.ir_version` field.
* Operators that are referenced by a given ONNX graph.  The set of ONNX operators in scope and their semantics is versioned via the *ONNX operator version*. The ONNX operator version is represented by the `ModelProto.op_version` field.
* An ONNX ModelProto that represents a given graph - that is, the contents of a model. We refer to this version as the *model version* and it is represented by the `ModelProto.model_version` field.

The versioning of all three of these entity types is distinct, and largely independent. That is, ONNX versions the IR format at a different rate than the operators defined by ONNX - the former will version much slower than the latter.

While the versioning mechanisms are clearly specified in this document, the policies for version management are only mandated for IR version and operator version.

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

ONNX serializes the MAJOR, MINOR, and PATCH values as a bit-packed 32-bit integer; the most siginificant byte is the MAJOR component, the second most significant byte is the MINOR component, the least significant two bytes are the PATCH component.

For example, 1.2.345 is represented as 0x01020159.

## IR versioning

Changes to the file format or graph model semantics version atomically. Breaking changes to the format or semantics of the ONNX specification require an increment of the MAJOR version.  Non-breaking format or semantic changes that introduce new functionality require an increment of the MINOR version. Non-breaking changes to the specification that simply clarify spec ambiguities require an increment of the PATCH version.

The ONNX IR format adheres to the versioning guidelines defined in the [Updating a Message Type](https://developers.google.com/protocol-buffers/docs/proto3#updating) section of the proto3 specification.

As a general principle, implementations SHOULD be robust in the face of missing fields. However, to ensure basic interoperation, a subset of message fields will be marked as required for a given IR version and all producers MUST set these fields correctly. Required fields always are marked with the comment:

    // This field MUST be present for this version of the IR.

By way of example, the `ModelProto.ir_version` MUST be present in every model.  The ONNX checker (`onnx/checker.py`) will enforce these rules.

ISSUE: decide and document how we want to handle changes to the format that are forward compatible but will cause the generated code to introduce a breaking change (e.g., changing the data type of a field from int64 to int32, int64 to double or bytes to string).  I would recommend that we never do it after we hit 1.0.0 - prior to that we should be fairly liberal with changes to size (e.g., `int32<>int16`, `float<>double`) and conservative with changes to value space (e.g., `integral<>floatingpoint`, `string<>int`, `scalar<>message`)

ISSUE: define type compatibility rules either here or under model versioning - probably here

## Operator versioning

ONNX's operator versioning has semantic meaning: the ONNX operator
version associated with a model controls the set of ONNX operators it
may use, and the semantic meaning of these operators.  Operator
versioning permits ONNX to evolve the semantics of the operators it
specifies in BC-breaking ways, while preserving backwards-compatibility
with older exported models (which pin the meaning of operators by
recording the operator version they target).

The ONNX operator version used by a model is recorded in
`ModelProto.op_version` and is a simple integer (i.e., it does *not*
use semantic versioning.)  Each operator version of ONNX is associated
with some machine-readable metadata which specifies the implementation
for every operator that is in scope.  When ONNX makes BC-breaking
changes to the semantics of operators (e.g., by changing the meaning
of an operator or removing an operator), it increments the ONNX
operator version.  Additions of new operators do NOT increment the
ONNX operator version (as they are not BC-breaking changes), nor
do additions of new optional attributes.

When understanding the semantics of an operator version, it is helpful
to distinguish an *operator implementation* from an operator name
(`op_type`).  An operator implementation specifies the actual semantics
of an operator (its type, the mathematical computation it performs on
its arguments, etc.); an operator name is simply what is recorded in an
exported model.  Operator implementations are *immutable*; they never
change across ONNX operator versions.  For example, if ONNX wants to
introduce a breaking change to the `Conv` operator, it must create a
new operator implementation describing the new desired behavior, and
then remap the `Conv` name to this new implementation in the next ONNX
operator version.  In this way, the ONNX operator version is a
*namespace* mechanism on top of an immutable store of operator
implementations; each operator version defines a mapping of operator
names to operator implementations.

It will be helpful to refer to operator implementations by name without
reference to an operator version. To distinguish these identifiers from
operator names, they have a leading underscore (e.g., the old and new
implementations of `Conv` might be `_Conv` and `_ConvWithBias`).

### Frontends export models targeting a specific operator version

The frontend of a given ONNX model MUST pick a specific ONNX operator
version they target; we refer to this as the *target ONNX operator
version*.  They MUST use only operators which are defined within this
version; the semantics of the operators are exactly as defined by that
ONNX operator version.

Upon serializing an ONNX model, the frontend SHOULD record in
`ModelProto.op_version` smallest operator version for which the used
subset of the mapping from operator names to operator implementations is
equal to the corresponding subset of the target ONNX operator version
mapping.  For example, if a frontend targets ONNX operator version 5,
but only uses operators which were available since operator version 3,
it should export a model with `op_version = 3`.
This ensures that exported models can be understood by
backends which don't understand later ONNX operator versions.

### Backends use the operator version database to pick implementations

A backend which interprets ONNX models MUST declare a maximum ONNX
operator versions it is aware of; this defines the *known ONNX operator
version range* (which always extends from the beginning of ONNX's
operator version history, to some particular operator version).  A
backend MUST *unconditionally* reject any models whose `op_version`
beyond this version.  ONNX may make BC-breaking changes to operators; a
backend which is not aware of these changes cannot know if they should
reject a model because it uses an operator that was changed in a
backwards incompatible way.

A backend MAY implement some of the operator implementations utilized by
this range (it is not expected to implement *all* operator
implementations, especially legacy ones; however, a larger set of
operator implementations means better interoperability).  For every ONNX
operator version inside the known ONNX operator version range, a backend
MUST be able to compute the corresponding mapping from operator name to
operator implementation.  To interpret a model, a backend MUST use the
recorded `ModelProto.op_version` to translate operator names `op_type`
into the actual implementations of the operators.

To allow backends to easily compute the operator name to implementation
mapping, ONNX publishes an *operator version database* in an auxiliary
protobuf format, which defines all of the name to implementation
mappings for all operator versions up to the latest operator version at
the time of the database's publication.  A backend SHOULD ship a copy of
the operator version database corresponding to the latest ONNX operator
version they support and use it to perform the mapping computation
(frontends which target a later ONNX operator version may still export
models which are interpretable by this backend, if they recorded a
minimum compatible `op_version`).  The operator version database
increases monotonically; mappings for old versions never change.
(Internally, the operator version database is represented as a
changelog, recording the deltas from one operator version to the next.)

### Vendor extensions to operators

We don't expect ONNX to cover every operator that every vendor may use;
some backends may need to expose specific operators before they
can go through the ONNX standardization process.  Thus, ONNX operator
names are extended with an optional namespace prefix, specified by an
alphanumeric string and a period, can place their custom operators
under.  For example, a Caffe2 only operator might be assigned the name
`Caffe2.ConvTBC`.  The list of vendor prefixes will be coordinated by
the ONNX standards team, in a manner similar to URI schemes.

Each vendor operator extension namespace can also have an operator
version number associated with it.  The operator versions for every
vendor extension used in a model are stored in a list of key-value pairs in
`ModelProto.ext_op_versions`; each operator version controls the
semantics of the operators in its indicated namespace.  Operator
implementations are similarly namespaced, e.g., `Caffe2._ConvTBCImpl`,
and each extension provider MUST provide an operator version database
for their names.

A backend MUST reject a model which uses an extension namespace that
it does not understand.

### The operator implementation database

The operator implementation database records information about operator
implementations, which frontends and backends MAY use to implement
facilities such as type checking imported models or determining
the portability/stability of an exported model.  The operator
implementation database is mapping from operator implementation names to
information about the operator, e.g., types, stability, documentation,
etc.  Use of the operator implementation database is OPTIONAL; a backend
may choose to hard code their information about implementations (as such
information is immutable.)

One thing to note is that the operator version database does NOT contain
the operator implementation database; it contains only the bare minimum
to calculate the mapping of operator names to operator implementations;
no more, no less.

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

ISSUE: what's our policy for accuracy or perf changes?
