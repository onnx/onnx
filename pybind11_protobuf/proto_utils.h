// Copyright (c) 2019 The Pybind Development Team. All rights reserved.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#ifndef PYBIND11_PROTOBUF_PROTO_UTILS_H_
#define PYBIND11_PROTOBUF_PROTO_UTILS_H_

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

#include <google/protobuf/any.pb.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
#include <google/protobuf/reflection.h>
//#include "absl/strings/string_view.h"

namespace pybind11 {
namespace google {

// Alias for checking whether a c++ type is a proto.
template <typename T>
static constexpr bool is_proto_v = std::is_base_of<::google::protobuf::Message, T>::value;

// Name of the property which indicates whether a proto is a wrapped or native.
constexpr char kIsWrappedCProtoAttr[] = "_is_wrapped_c_proto";

// Returns true if the given python object is a wrapped C proto.
inline bool IsWrappedCProto(handle handle) {
  return hasattr(handle, kIsWrappedCProtoAttr);
}

// Gets the field with the given name from the given message as a python object.
object ProtoGetField(::google::protobuf::Message* message, absl::string_view name);
object ProtoGetField(::google::protobuf::Message* message,
                     const ::google::protobuf::FieldDescriptor* field_desc);

// Sets the field with the given name in the given message from a python object.
// As in the native API, message, repeated, and map fields cannot be set.
void ProtoSetField(::google::protobuf::Message* message, absl::string_view name,
                   handle value);
void ProtoSetField(::google::protobuf::Message* message,
                   const ::google::protobuf::FieldDescriptor* field_desc, handle value);

// Initializes the fields in the given message from the the keyword args.
// Unlike ProtoSetField, this allows setting message, map and repeated fields.
void ProtoInitFields(::google::protobuf::Message* message, kwargs kwargs_in);

// Wrapper around ::google::protobuf::Message::CopyFrom which can efficiently copy from
// either a wrapped C++ or native python proto. Throws an error if `other`
// is not a proto of the correct type.
void ProtoCopyFrom(::google::protobuf::Message* msg, handle other);

// Wrapper around ::google::protobuf::Message::MergeFrom which can efficiently merge from
// either a wrapped C++ or native python proto. Throws an error if `other`
// is not a proto of the correct type.
void ProtoMergeFrom(::google::protobuf::Message* msg, handle other);

// If py_proto is a native c or wrapped python proto, sets name and returns
// true. If py_proto is not a proto, returns false.
bool PyProtoFullName(handle py_proto, std::string* name);

// Returns whether py_proto is a proto and matches the expected_type.
bool PyProtoCheckType(handle py_proto, const std::string& expected_type);
// Throws a type error if py_proto is not a proto or the wrong message type.
void PyProtoCheckTypeOrThrow(handle py_proto, const std::string& expected_type);

// Returns whether py_proto is a proto and matches the ProtoType.
template <typename ProtoType>
bool PyProtoCheckType(handle py_proto) {
  return PyProtoCheckType(py_proto, ProtoType::descriptor()->full_name());
}

// Returns whether py_proto is a proto based on whether it has a descriptor
// with the name of the proto.
template <>
inline bool PyProtoCheckType<::google::protobuf::Message>(handle py_proto) {
  return PyProtoFullName(py_proto, nullptr);
}

// Returns the serialized version of the given (native or wrapped) python proto.
std::string PyProtoSerializeToString(handle py_proto);

// Gets the descriptor for the proto specified by the argument, which can be a
// native python proto, a wrapped C proto, or a string with the full type name.
const ::google::protobuf::Descriptor* PyProtoGetDescriptor(handle py_proto);

// Allocate and return the ProtoType given by the template argument.
// py_proto is not used in this version, but is used by a specialization below.
template <typename ProtoType>
std::unique_ptr<ProtoType> PyProtoAllocateMessage(handle py_proto = handle(),
                                                  kwargs kwargs_in = kwargs()) {
  auto message = std::make_unique<ProtoType>();
  ProtoInitFields(message.get(), kwargs_in);
  return message;
}

// Specialization for the case that the template argument is a generic message.
// The type is pulled from the py_proto, which can be a native python proto,
// a wrapped C proto, or a string with the full type name.
template <>
std::unique_ptr<::google::protobuf::Message> PyProtoAllocateMessage(handle py_proto,
                                                        kwargs kwargs_in);

// Allocate and return a message instance for the given descriptor.
std::unique_ptr<::google::protobuf::Message> PyProtoAllocateMessage(
    const ::google::protobuf::Descriptor* descriptor, kwargs kwargs_in);

// Allocate a C++ proto of the same type as py_proto and copy the contents
// from py_proto. This works whether py_proto is a native or wrapped proto.
// If expected_type is given and the type in py_proto does not match it, an
// invalid argument error will be thrown.
template <typename ProtoType>
std::unique_ptr<ProtoType> PyProtoAllocateAndCopyMessage(handle py_proto) {
  auto new_msg = PyProtoAllocateMessage<ProtoType>(py_proto);
  ProtoCopyFrom(new_msg.get(), py_proto);
  return new_msg;
}

// Pack an any proto from a proto, regardless of whether it is a native python
// or wrapped c proto. Using the converter on a native python proto would
// require serializing-deserializing-serializing again, while this always
// requires only 1 serialization operation. Returns true on success.
bool AnyPackFromPyProto(handle py_proto, ::google::protobuf::Any* any_proto);

// Unpack the given any proto into the given py_proto, regardless of whether it
// is a native python or wrapped c proto. Returns true on success.
bool AnyUnpackToPyProto(const ::google::protobuf::Any& any_proto,
                        handle py_proto);

// Returns the pickle bindings (passed to class_::def) for ProtoType.
template <typename ProtoType>
decltype(auto) MakePickler() {
  return pybind11::pickle(
      [](ProtoType* message) {
        return dict("serialized"_a = bytes(message->SerializeAsString()),
                    "type_name"_a = message->GetTypeName());
      },
      [](dict d) {
        auto message = PyProtoAllocateMessage<ProtoType>(d["type_name"]);
        // TODO(b/145925674): skip creation of temporary string once bytes
        // supports string_view casting.
        message->ParseFromString(std::string(d["serialized"].cast<bytes>()));
        return message;
      });
}

// Add bindings for the given concrete ProtoType, returning the class_ instance
// so that type-specific bindings can be added. This should only be used with
// well known types (google3/net/proto2/python/internal/well_known_types.py);
// all other message types should use RegisterProtoMessageType.
template <typename ProtoType>
auto ConcreteProtoMessageBindings(handle module) {
  // Register the type.
  auto* descriptor = ProtoType::GetDescriptor();
  const char* registered_name = descriptor->name().c_str();
  class_<ProtoType, ::google::protobuf::Message, std::shared_ptr<ProtoType>> message_c(
      module, registered_name);
  // Add a constructor.
  message_c.def(init([](kwargs kwargs_in) {
    return PyProtoAllocateMessage<ProtoType>(handle(), kwargs_in);
  }));
  // Create a pickler for this type (the base class pickler will fail).
  message_c.def(MakePickler<ProtoType>());
  message_c.def("__copy__", [](object self) {
      return PyProtoAllocateAndCopyMessage<ProtoType>(self);
      });
  message_c.def("__deepcopy__", [](object self, dict) {
      return PyProtoAllocateAndCopyMessage<ProtoType>(self);
      }, arg("memo"));
  // Add the descriptor as a static field.
  message_c.def_readonly_static("DESCRIPTOR", descriptor);
  // Add bindings for each field, to avoid the FindFieldByName lookup.
  for (int i = 0; i < descriptor->field_count(); ++i) {
    auto* field_desc = descriptor->field(i);
    message_c.def_property(
        field_desc->name().c_str(),
        [field_desc](::google::protobuf::Message* message) {
          return ProtoGetField(message, field_desc);
        },
        [field_desc](::google::protobuf::Message* message, handle value) {
          ProtoSetField(message, field_desc, value);
        });
  }
  return message_c;
}

// Registers the bindings for the proto base types in the given module. Can only
// be called once; subsequent calls will fail due to duplicate registrations.
void RegisterProtoBindings(module m);

// If modifying the functions below, see
// g3doc/pybind11_protobuf/README.md#importing-the-proto-module

// Returns true if the proto module has been imported.
inline bool IsProtoModuleImported() {
  return detail::get_type_info(typeid(::google::protobuf::Message));
}

// In debug mode, throws a type error if the proto module is not imported.
// No-opt if NDEBUG is defined, and inlined so the compiler can optimize it out.
inline void CheckProtoModuleImported() {
#ifndef NDEBUG
  if (!IsProtoModuleImported())
    throw type_error(
        "Proto module has not been imported. Did you call ::pybind11::google"
        "::ImportProtoModule() in your PYBIND11_MODULE definition?");
#endif
}

}  // namespace google
}  // namespace pybind11

#endif  // PYBIND11_PROTOBUF_PROTO_UTILS_H_
