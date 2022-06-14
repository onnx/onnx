#ifndef PYBIND11_PROTOBUF_PROTO_CAST_UTIL_H_
#define PYBIND11_PROTOBUF_PROTO_CAST_UTIL_H_

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>

namespace pybind11_protobuf {

// Initialize internal proto cast dependencies, which includes importing
// various protobuf-related modules.
void InitializePybindProtoCastUtil();

// Imports a module pertaining to a given ::google::protobuf::Descriptor, if possible.
void ImportProtoDescriptorModule(const ::google::protobuf::Descriptor *);

// Returns a ::google::protobuf::Message* from a cpp_fast_proto, if backed by C++.
const ::google::protobuf::Message *PyProtoGetCppMessagePointer(pybind11::handle src);

// Returns the protocol buffer's py_proto.DESCRIPTOR.full_name attribute.
std::string PyProtoDescriptorName(pybind11::handle py_proto);

// Return whether py_proto is compatible with the C++ descriptor.
// The py_proto name must match the C++ Descriptor::full_name(), and is
// expected to originate from the python default pool, which means that
// this method will return false for dynamic protos.
bool PyProtoIsCompatible(pybind11::handle py_proto,
                         const ::google::protobuf::Descriptor *descriptor);

// Allocates a C++ protocol buffer for a given name.
std::unique_ptr<::google::protobuf::Message> AllocateCProtoFromPythonSymbolDatabase(
    pybind11::handle src, const std::string &full_name);

// Serialize the py_proto and deserialize it into the provided message.
// Caller should enforce any type identity that is required.
bool PyProtoCopyToCProto(pybind11::handle py_proto, ::google::protobuf::Message *message);
void CProtoCopyToPyProto(::google::protobuf::Message *message, pybind11::handle py_proto);

// Returns a handle to a python protobuf suitably
pybind11::handle GenericFastCppProtoCast(::google::protobuf::Message *src,
                                         pybind11::return_value_policy policy,
                                         pybind11::handle parent,
                                         bool is_const);

pybind11::handle GenericPyProtoCast(::google::protobuf::Message *src,
                                    pybind11::return_value_policy policy,
                                    pybind11::handle parent, bool is_const);

pybind11::handle GenericProtoCast(::google::protobuf::Message *src,
                                  pybind11::return_value_policy policy,
                                  pybind11::handle parent, bool is_const);

}  // namespace pybind11_protobuf

#endif  // PYBIND11_PROTOBUF_PROTO_CAST_UTIL_H_
