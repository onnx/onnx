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
* The parallelized serialization is not yet implemented but that's an easy fix.
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

## Protobuf format

Let's dig into the class OperatorSetIdProto with contains an integer (version) and a string (domain).

```
message OperatorSetIdProto {
  optional string domain = 1;
  optional int64 version = 2;
}
```

Protobuf serializes this structure by aggregating the following information:

- a varint containing the field number for the domain and its type,
  because it is stored as a buffer of a fixed size (see function ``write_field_header``)
- the size of the string
- the string
- again a varint the field number for the version and its type, variable,
  because it is stored as a varint.

A varint is a variable integer (see [Encoding](https://protobuf.dev/programming-guides/encoding/),
any LLM usually gives a good implementation on how to encode or decode such a number,
see function ``write_variant_uint64``).

A varint or variable int is an ``uint64_t`` written with between 1 or 10 bytes.
If it is in 0-127, then it is 1 byte, otherwise, at least two bytes are used.
So a varint is represented as a the shortest sequence of groups of 7 bits.

A more complex structure is stored as a sequence of the three information:

- a varint containing the field number for the domain and its type (fixed)
- the length of the buffer containing the length for the serialized attribute,
  unless it is a standard numerical type
- the buffer with the serialized attribute

Protobuf is backward compatible in a way that any new attribute receives a new field number.
An old version of protobuf can read a new format by ignoring the new type.
A new version of protobuf can write an old format by ignoring the deprecated type.
The compatibility is maintained as long as no old field number is reused for another
type.

Reading protobuf is very fast and the structure can be created while reading the serialized string.
Writing protobuf is more complex. We need to serialize an attribute to know how many bytes
it takes on disk. When it is serialized, the length of the buffer is known and can be stored as a
variable integer. So we need to know the size in order to know how many bytes are needed to store
the size and the buffer of any object. That's why the current implementation serializes a class
in two steps.

The first one consists in computing the size of every object and nested object.
It caches them into the stream doing the serialization.
Once it is done, we know where exactly every serialized object must be copied into
the final string. That what allows us to efficiently parallelize the serialization
of onnx models because most of the space is taken by tensors, easy to serialize.

The second step consists in doing the serialization. It does not need any additional space
in memory to store the serialized nested object: their size is already known.

## Script to measure the loading time

The following script compares the python bindings of onnx and onnx2
to load and save a model, with or without external data, with or
without parallelization.

```python
"""
Measures loading, saving time for an onnx model in python
=========================================================


The script creates an ONNX model and measures the time to load and save it
with onnx and onnx2. This only compares the python bindings.
"""

import os
import time
import numpy as np
import pandas
import onnx
import onnx.onnx2 as onnx2


data = []

# An example.
onnx_file = "microsoft_Phi-4-mini-reasoning-onnx-dynamo-ir.onnx"
onnx_data = onnx_file + ".data"
if not os.path.exists(onnx_file):
    print("Creates the model, starts with importing transformers...")
    # Your implementation here.
    print("done.")

# %%
# Let's load and save the model to get one unique file.

full_name = "microsoft_Phi-4-mini-reasoning.onnx"
if not os.path.exists(full_name):
    print("Loads the model and saves it as one unique file.")
    onx = onnx.load(onnx_file)
    onnx.save(onx, full_name)
    print("done.")

# %%
# Let's get the size.


size = os.stat(full_name).st_size
print(f"model size {size / 2**20:1.3f} Mb")

# %%
# Measures the loading time
# +++++++++++++++++++++++++


def measure(step_name, f, N=3):
    times = []
    for _ in range(N):
        begin = time.perf_counter()
        onx = f()
        end = time.perf_counter()
        times.append(end - begin)
    res = {"avg": np.mean(times), "times": times}
    data.append(
        dict(name=step_name, avg=res["avg"], min=np.min(times), max=np.max(times))
    )
    return onx, res


# %%
# Let's do it with onnx2.

print("Loading time with onnx2.")
onx2, times = measure("load/onnx2", lambda: onnx2.load(full_name))
print(times)

# %%
# Then with onnx.

print("Loading time with onnx.")
onx, times = measure("load/onnx", lambda: onnx.load(full_name))
print(times)

# %%
# Let's do it with onnx2 but the loading of the tensors is parallelized.

print(
    f"Loading time with onnx2 and 4 threads, "
    f"it has {len(onx2.graph.initializer)} initializers"
)
onx2, times = measure(
    "load/onnx2/x4", lambda: onnx2.load(full_name, parallel=True, num_threads=4)
)
print(times)

# %%
# It looks much faster.

# %%
# Let's load it with :epkg:`onnxruntime`.
import onnxruntime  # noqa: E402

so = onnxruntime.SessionOptions()
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
print("Loading time with onnxruntime")
_, times = measure(
    "load/ort",
    lambda: onnxruntime.InferenceSession(
        full_name, so, providers=["CPUExecutionProvider"]
    ),
)
print(times)


# %%
# Measure the saving time
# +++++++++++++++++++++++
#
# Let's do it with onnx2.

print("Saving time with onnx2.")
_, times = measure("save/onnx2", lambda: onnx2.save(onx2, full_name))
print(times)

# %%
# Then with onnx.

print("Saving time with onnx.")
_, times = measure("save/onnx", lambda: onnx.save(onx, full_name))
print(times)

# %%
# Measure the saving time with external weights
# +++++++++++++++++++++++++++++++++++++++++++++
#
# Let's do it with onnx2.

full_name = "dump_test/microsoft_Phi-4-mini-reasoning.ext.onnx"
full_weight = "dump_test/microsoft_Phi-4-mini-reasoning.ext.data"

print("Saving time with onnx2 and external weights.")
_, times = measure(
    "save/onnx2/ext", lambda: onnx2.save(onx2, full_name, location=full_weight)
)
print(times)

# %%
# Then with onnx. We can only do that once,
# the function modifies the model inplace to add information
# about external data. The second run does not follow the same steps.

print("Saving time with onnx and external weights.")
full_weight += ".2"
_, times = measure(
    "save/onnx/ext",
    lambda: onnx.save(
        onx,
        full_name,
        location=os.path.split(full_weight)[-1],
        save_as_external_data=True,
        all_tensors_to_one_file=True,
    ),
    N=1,
)
print(times)

# %%
# Measure the load time with external weights
# +++++++++++++++++++++++++++++++++++++++++++
#
# Let's do it with onnx2.

print("Loading time with onnx2 and external weights.")
_, times = measure("load/onnx2/ext", lambda: onnx2.load(onnx_file, location=onnx_data))
print(times)

# %%
# Same measure but parallelized.

print("Loading time with onnx2 parallelized and external weights.")
_, times = measure(
    "load/onnx2/ext/x4",
    lambda: onnx2.load(onnx_file, location=onnx_data, parallel=True, num_threads=4),
)
print(times)

# Let's do it with onnx2.

print("Saving time with onnx and external weights.")
_, times = measure("load/onnx/ext", lambda: onnx.load(onnx_file))
print(times)


# %%
# Plots
# +++++

df = pandas.DataFrame(data).sort_values("name").set_index("name")
print(df)

# %%
# Visually.

ax = df[["avg"]].plot.barh(
    title=f"size={size / 2**20:1.3f} Mb\n"
    "onnx VS onnx2 for load/save (s)\nthe lower, "
    "the better\next = external data\nx4 = 4 threads"
)
ax.figure.tight_layout()
ax.figure.savefig("plot_onnx2_time.png")
```