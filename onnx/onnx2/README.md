# ONNX2: ONNX without protobuf

onnx2 is a prototype which replicates onnx API without using protobuf.
It replaces the protobuf definition of ONNX classes (ModelProto, TensorProto, ...).
It is able to load and save onnx files with the same format as protobuf.

```python
import onnx.onnx2 as onnx2

proto = onnx2.load("filename.onnx")
assert isinstance(proto, onnx2.ModelProto)
print(f"The model has {len(proto.graph.node)} nodes.")
```

The code is not perfect and can be surely improved but it already
supports the serialization and parsing for all onnx classes (except TrainingInfoProto and Opaque).
It is concrete enough to open a discussion on whether this could
replace protobuf. Many tests were implemented to make sure
this can be done (see tests in ``onnx/tests/test_onnx2_*`` and ``onnx/tests/cpp/test_onnx2_*``).
One file goes through all onnx files in the backend test and makes
sure the model remains the same after it was deserialized and serialized again
with onnx2. Every model is loaded with onnx2,
serialized with onnx2 then parsed with onnx and serialized with onnx.
The final serialized strings is equal to the original one.
Currently, the serialized strings have the same content, only
the order of the attribute may change.

It currently provides the following features:

* Compatibility with protobuf format.
* No dependency on protobuf, so it should make any build easier.
* No need to serialize/deserialize when switching from python to C++ and the way around.
* No limitation on file size (2 Gb for protobuf).
  The code has yet to be tested to support tensors bigger than 2 Gb.
* Writing / Reading file with external data is directly handled in C++.
* The parsing a file can be called parallelized. When the parser walks through the file, 
  it skips every big tensor and pushes the loading of the field ``raw_data``
  in a queue. This queue is processed by as many threads as the user wants.
  This is the only implemented strategy but with 4 threads, a 1Gb model can be read twice faster.
* The current API in python is very close to the existing ones,
  functions in ``onnx.helper.py`` runs without any change.
* It is possible to skip the loading of the weights of a tensor even if the model
  was stored in one unique file (no external weight).

The features which are not implemented yet:

* TrainingInfoProto and Opaque, easy to add, skipped because rarely used
* The python API still misses some unfrequent functions such as ``HasField`` (easy to fix).
* The serialization is not yet implemented but that's an easy fix.
* The C++ API is close to the existing one but this was not tested yet.
* ``oneof`` rule: the implementation is not complete yet and could be improvoed.
  Right now, every *oneof* attribute does not share the same space in memory.

The features we could easily implement:

* Since the library implements the parsing of a file or a string,
  we could easily implements a way to load the weights directly wherever
  the user wants them to be and avoid sometimes an unnecessary copy.
* New serialization format: it could be done elsewhere but this work offers
  a quick start easier to leverage.
* Partial loading or writing of models, to load different parts of
  a model on different machines for example. Even though a file has to be 
  read entirely, it is possible and easy to quickly go through the desired
  parts without parsing too much of the head of the file.
* Research around a more compressed format (detects duplication of strings, 
  compressed strings, ...)
* ...

## Proto Definition

Their definition can be found in [onnx/onnx2/cpu/onnx2.h](onnx/onnx2/cpu/onnx2.h).
Here is a simple example for class ``OperatorSetIdProto``.
The current implementation in C++ uses many macros instead of templates.
That's something which could be easily improved.

```python
BEGIN_PROTO(OperatorSetIdProto, "Defines a unqiue pair domain, opset version for a set of operators.")
FIELD_STR(
    domain,
    1,
    "The domain of the operator set being identified. The empty string ("
    ") or absence of this field implies the operator set that is defined as part of the "
    "ONNX specification. This field MUST be present in this version of the IR when "
    "referring to any other operator set.")
FIELD_DEFAULT(
    int64_t,
    version,
    2,
    0,
    "The version of the operator set being identified. This field MUST be present in "
    "this version of the IR.")
END_PROTO()
```

Their implementation is done in [onnx/onnx2/cpu/onnx2.cpp](onnx/onnx2/cpu/onnx2.cpp).
It should be possible to refactor the code to avoid writing the list of all attributes
four times:

* one to serialize
* one to parse
* one to compute the size of the serialized objects (this is important to speed up the writing),
  this feature alone allows any future parallelization of the serialization.
* one to print: this one could really be improved, it was more a way to experiment
  an implementation with templates and not macros.

```python
IMPLEMENT_PROTO(OperatorSetIdProto)
uint64_t OperatorSetIdProto::SerializeSize(utils::BinaryWriteStream& stream, SerializeOptions& options) const {
  uint64_t size = 0;
  SIZE_FIELD_EMPTY(size, options, stream, domain)
  SIZE_FIELD(size, options, stream, version)
  return size;
}
void OperatorSetIdProto::SerializeToStream(utils::BinaryWriteStream& stream, SerializeOptions& options) const {
  WRITE_FIELD_EMPTY(options, stream, domain)
  WRITE_FIELD(options, stream, version)
}
void OperatorSetIdProto::ParseFromStream(utils::BinaryStream& stream, ParseOptions& options){
    READ_BEGIN(options, stream, OperatorSetIdProto)
    READ_FIELD(options, stream, domain)
    READ_FIELD(options, stream, version)
    READ_END(options, stream, OperatorSetIdProto)
}
std::vector<std::string> OperatorSetIdProto::PrintToVectorString(utils::PrintOptions& options) const {
  return write_proto_into_vector_string(options, NAME_EXIST_VALUE(domain), NAME_EXIST_VALUE(version));
}
```

Everything not related to the onnx class definition was placed in other
files. This work would be used to define other classes than the ONNX ones.

## Misc

**Optional, Repeated Fields**

``std::string`` was replaced by a custom implementation not stored the final 0
in the protos. A template ``RepeatedField`` implements the repeated fields
from protobuf, ``OptionalField`` does the same for optional. Except for strings,
there is no optional strings.

**Empty, Null Strings**

Protobuf does make the difference between an empty string ``""`` and a null pointer.
The library does the same if explicitly told (look for macro ``WRITE_FIELD_EMPTY``).
That's something which could be dropped.

**External Data**

When writing a model with external data (weights are stored in a separate files),
the model is temporarily modified to append the external data.
Once the model is saved, this information is removed and the model
restored to its original state.

However, this information is not really necessary as long as the model
stores the weights in the same order.
