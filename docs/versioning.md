# ONNX versioning

This document describes the rules for versioning ONNX. Like the rest of the ONNX
specification, MUST, SHOULD et al are used consistent with [RFC2119](https://tools.ietf.org/html/rfc2119).  

## Versioning Principles

ONNX defines versioning policy and mechanism for three classes of entities:

* The abstract graph model and the concrete format that serializes it. The graph model and format are versioned atomically and are referred to as the *IR version.* The IR version is  represented by the `ModelProto.ir_version` field.  
* Operators that are referenced by a given ONNX graph. The version of a given operator  is referred to as the *operator version*. The operator version is  represented by the `TBD` field.  
* An ONNX ModelProto that represents a given graph - that is, the contents of a model. We refer to this version as the *model version* and it is represented by the `ModelProto.model_version` field.    

The versioning of all three of these entity types is distinct, and largely independent. That is, ONNX versions the IR format at a different rate than the operators defined by ONNX - the former will version much slower than the latter. 

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

Per the rules of SemVer 2.0, during the initial development of ONNX:
* We use a MAJOR version of 0 for both the IR version and operator version.
* We will only increment the MINOR version in the face of either a breaking change as defined in this specification or the need to stabilize a version for specific engineering purposes.

Once we declare a stable/released version of ONNX (e.g., we hit 1.0.0), we will adhere to the standard SemVer rules for versioning.

 [ISSUE: decide how we will stabilize and archive released versions of ONNX] 

### Serializing SemVer version numbers in protobuf

For historical reasons, ONNX serializes the MAJOR, MINOR, and PATCH values as a bit-packed 32-bit integer; the most siginificant byte is the MAJOR component, the second most significant byte is the MINOR component, the least significant two bytes are the PATCH component. 

For example, 1.2.345 is represented as 0x01020159.

The pre-release and build metadata are represented as ISSUE: are we supporting these or not, and if so, on what entity types? 





## IR versioning

Changes to the file format or graph model semantics version atomically. Breaking changes to the format or semantics of the ONNX specification require an increment of the MAJOR version.  Non-breaking format or semantic changes that introduce new functionality require an increment of the MINOR version. Non-breaking changes to the specification that simply clarify spec ambiguities require an increment of the PATCH version.  

The ONNX IR format adheres to the versioning guidelines defined in the [Updating a Message Type](https://developers.google.com/protocol-buffers/docs/proto3#updating) section of the proto3 specification.  

As a general principle, implementations SHOULD be robust in the face of missing fields. However, to ensure basic interoperation, a subset of message fields will be marked as required for a given IR version and all producers MUST set these fields correctly. Required fields always are marked with the comment:

    // This field MUST be present for this version of the IR.

By way of example, the ModelProto.ir_version MUST be present in every model.  The ONNX checker (onnx/checker.py) will enforce these rules.

ISSUE: decide and document how we want to handle changes to the format that are forward compatible but will cause the generated code to introduce a breaking change (e.g., changing the data type of a field from int64 to int32, int64 to double or bytes to string).  I would recommend that we never do it after we hit 1.0.0 - prior to that we should be fairly liberal with changes to size (e.g., int32<>int16, float<>double) and conservative with changes to value space (e.g., integral<>floatingpoint, string<>int, scalar<>message)

ISSUE: define type compatiblity rules either here or under model versioning - probably here

## Operator versioning

ISSUE: sort out the design and then document.

## Model versioning

Model versioning is ultimately the domain of a given organization, and therefor, this section of the spec is non-normative and simply proposes a set of practices to consider.

Model authors and applications/systems  MAY elect to ignore model versioning mechanism and policy rules. For models that will be shared across developers, teams, or organizations, model authors and applications/systems SHOULD adhere to the following version policies:

ISSUE: the following is a strawman. I'm confident some of it is right and some is wrong. Either way, we need to make some calls and document it.

### Signature Changes
1. Breaking changes to the ModelProto.graph.GraphProto.input or .output MUST increment the MAJOR version of ModelProto.model_version. Breaking changes include:
    
    * Breaking changes to the semantics of an input or output (e.g., changing the required contents of an input tensor from color image to black and white image). 
    * Changing the declared type of an input or output to an incompatible type (e.g., tensor(int)->tensor(string).
    * Adding a new input for which there is no meaningful or specified default value. For graph inputs, those values are provided by a same-named value in GraphProto. initializer.
    * Removing an existing output for which there is no meaningful or specified default value.

2. Non-breaking changes to the ModelProto.graph.GraphProto.input or .output MUST increment the MINOR version of ModelProto.model_version. Non-breaking changes include:
    
    * Changing the declared type of an input or output to an compatible/widening type (e.g., tensor(int32)->tensor(int64), tensor(float16)->tensor(float32).
    * Adding a new input for which there is a meaningful or specified default value.
    * Adding new behavior that is only triggered in the presence of inputs that were not
    possible in prior versions of the graph (typically by the presense of a new input 
    or allowing a previously invalid input value).

### IR version/Operator version dependency changes

ISSUE: what's our policy when a model takes a dependency on new IR_VERSION change and/or new operator change?


### Accuracy or performance changes

Assuming that there are no breaking changes to the signature of the model's graph or any operator dependencies, the shape and contents of the graph can change freely provided there are no semantic changes to the model. However, changes to the shape and contents of the graph can impact model accuracy and/or model performance.    

ISSUE: what's our policy for accuracy or perf changes?
