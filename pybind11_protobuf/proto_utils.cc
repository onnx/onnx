// Copyright (c) 2019 The Pybind Development Team. All rights reserved.
//
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "proto_utils.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <typeindex>

#include <google/protobuf/any.pb.h>
#include <google/protobuf/descriptor.pb.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
//#include "absl/strings/string_view.h"

namespace pybind11 {
namespace google {
namespace {

// A type used with DispatchFieldDescriptor to represent a generic enum value.
struct GenericEnum {};

// Calls Handler<Type>::HandleField(field_desc, ...) with the Type that
// corresponds to the value of the cpp_type enum in the field descriptor.
// Returns the result of that function. The return type of that function
// cannot depend on the Type template parameter (ie, all instantiations of
// Handler must return the same type).
//
// This works by instantiating a template class for each possible type of proto
// field. That template class is a template parameter (look up 'template
// template class'). Ideally this would be a template function, but template
// template functions don't exist in C++, so you must define a class with a
// static member function (HandleField) which will be called.
template <template <typename> class Handler, typename... Args>
auto DispatchFieldDescriptor(const ::google::protobuf::FieldDescriptor* field_desc,
                             Args... args)
    -> decltype(Handler<int32_t>::HandleField(field_desc, args...)) {
  // If this field is a map, the field_desc always describes a message with
  // 2 fields: key and value. In that case, dispatch based on the value type.
  // In all other cases, the value type is the same as the field type.
  const ::google::protobuf::FieldDescriptor* field_value_desc = field_desc;
  if (field_desc->is_map())
    field_value_desc = field_desc->message_type()->FindFieldByName("value");
  switch (field_value_desc->cpp_type()) {
    case ::google::protobuf::FieldDescriptor::CPPTYPE_INT32:
      return Handler<int32_t>::HandleField(field_desc, args...);
    case ::google::protobuf::FieldDescriptor::CPPTYPE_INT64:
      return Handler<int64_t>::HandleField(field_desc, args...);
    case ::google::protobuf::FieldDescriptor::CPPTYPE_UINT32:
      return Handler<uint32_t>::HandleField(field_desc, args...);
    case ::google::protobuf::FieldDescriptor::CPPTYPE_UINT64:
      return Handler<uint64_t>::HandleField(field_desc, args...);
    case ::google::protobuf::FieldDescriptor::CPPTYPE_FLOAT:
      return Handler<float>::HandleField(field_desc, args...);
    case ::google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE:
      return Handler<double>::HandleField(field_desc, args...);
    case ::google::protobuf::FieldDescriptor::CPPTYPE_BOOL:
      return Handler<bool>::HandleField(field_desc, args...);
    case ::google::protobuf::FieldDescriptor::CPPTYPE_STRING:
      return Handler<std::string>::HandleField(field_desc, args...);
    case ::google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE:
      return Handler<::google::protobuf::Message>::HandleField(field_desc, args...);
    case ::google::protobuf::FieldDescriptor::CPPTYPE_ENUM:
      return Handler<GenericEnum>::HandleField(field_desc, args...);
    default:
      throw std::runtime_error("Unknown cpp_type: " +
                               std::to_string(field_desc->cpp_type()));
  }
}

// Calls pybind11::cast, but change the exception type to type_error.
template <typename T>
T CastOrTypeError(handle arg) {
  try {
    return cast<T>(arg);
  } catch (const cast_error& e) {
    throw type_error(e.what());
  }
}

// Base class for type-agnostic functionality and data shared between all
// specializations of the ProtoFieldContainer (see below).
class ProtoFieldContainerBase {
 public:
  // Accesses the field indicated by `field_desc` in the given `proto`. `parent`
  // should be passed if and only if `proto` is a map key-value pair message.
  // In that case, it should point to the message which contains the map field.
  // (ie, the parent of the key-value pair). In all other cases, `proto` is
  // also the parent, which is indicated by passing nullptr for `parent`.
  ProtoFieldContainerBase(::google::protobuf::Message* proto,
                          const ::google::protobuf::FieldDescriptor* field_desc,
                          ::google::protobuf::Message* parent = nullptr)
      : proto_(proto),
        parent_(parent ? parent : proto),
        field_desc_(field_desc),
        reflection_(proto->GetReflection()) {}

  // Return the size of a repeated field.
  int Size() const { return reflection_->FieldSize(*proto_, field_desc_); }

  // Clear the field.
  void Clear() { this->reflection_->ClearField(proto_, field_desc_); }

  // Add is only available for embedded message fields; abort for all others.
  void Add() { std::abort(); }

 protected:
  // Throws an exception if the index is bad, adjusting for negative indexes
  // (which are relative to the end of the list). Returns the adjusted index.
  int CheckIndex(int idx) const {
    int size = Size();
    if (idx < 0) idx += size;  // Negative numbers index from the end.
    if (idx < 0 || idx >= size) {
      // This is faster than throwing a std::out_of_range exception.
      PyErr_SetString(PyExc_IndexError, "list index out of range");
      throw error_already_set();
    }
    return idx;
  }

  // Adjusts the index for negative values and clamps it to 0 or Size, which is
  // the standard logic for inserting list values. Returns the adjusted index.
  int ClampIndex(int idx) const {
    int size = Size();
    if (idx < 0) idx += size;  // Negative numbers index from the end.
    if (idx < 0) idx = 0;
    if (idx > size) idx = size;
    return idx;
  }

  // Cast `inst` and keep its parent alive until `inst` is no longer referenced.
  // Use this when (and only when) casting message, map and repeated fields.
  template <typename T>
  object CastAndKeepAlive(T* inst, return_value_policy policy) const {
    object py_inst = cast(inst, policy);
    // parent_ should be owned by a python object, and it's important to pass
    // the owning python object to keep_alive_impl. If a new python object is
    // created here, the original one would not be kept alive by py_inst and
    // memory could be freed too early. Cast uses a registry which maps C++
    // pointers to their corresponding python objects to return the owning
    // python object rather than creating a new one.
    object py_parent = cast(parent_, return_value_policy::reference);
    detail::keep_alive_impl(py_inst, py_parent);
    return py_inst;
  }

  ::google::protobuf::Message* proto_;
  ::google::protobuf::Message* parent_;
  const ::google::protobuf::FieldDescriptor* field_desc_;
  const ::google::protobuf::Reflection* reflection_;
};

// Create a template-specialization based reflection interface,
// which can be used with template meta-programming.
//
// Proto ProtoFieldContainer should be the only thing in this file that
// that directly uses the native reflection interface. All other accesses
// into the proto should go through the ProtoFieldContainer.
//
// Specializations should implement the following functions:
// cpp_type Get(int idx) const: Returns the value of the field, at the given
//   index if it is a repeated field (idx is ignored for singular fields).
// object GetPython(int idx) const: Converts the return value of
//   of Get(idx) to a python object.
// void Set(int idx, cpp_type value): Sets the value of the field, at the given
//   index if it is a repeated field (idx is ignored for singular fields).
// void SetPython(int idx, handle value): Sets the value of the field from the
//   given python value.
// void Append(handle value): Converts and adds the value to a repeated field.
// std::string ElementRepr(int idx) const: Convert the element to a string.
//
// Note: cpp_type may not be exactly the same as the template argument type-
// it could be a reference or a pointer to that type. Use ProtoFieldAccess<T>
// To get the exact type that is returned by the Get() method.
template <typename T>
class ProtoFieldContainer {};

// Create specializations of that template class for each numeric type.
// Unfortunately the type name is in the function names used to access the
// field, so the only way to do this is with a macro.
#define NUMERIC_FIELD_REFLECTION_SPECIALIZATION(func_type, cpp_type)     \
  template <>                                                            \
  class ProtoFieldContainer<cpp_type> : public ProtoFieldContainerBase { \
   public:                                                               \
    using ProtoFieldContainerBase::ProtoFieldContainerBase;              \
    cpp_type Get(int idx) const {                                        \
      if (field_desc_->is_repeated()) {                                  \
        return reflection_->GetRepeated##func_type(*proto_, field_desc_, \
                                                   CheckIndex(idx));     \
      } else {                                                           \
        return reflection_->Get##func_type(*proto_, field_desc_);        \
      }                                                                  \
    }                                                                    \
    object GetPython(int idx) const { return cast(Get(idx)); }           \
    void Set(int idx, cpp_type value) {                                  \
      if (field_desc_->is_repeated()) {                                  \
        reflection_->SetRepeated##func_type(proto_, field_desc_,         \
                                            CheckIndex(idx), value);     \
      } else {                                                           \
        reflection_->Set##func_type(proto_, field_desc_, value);         \
      }                                                                  \
    }                                                                    \
    void SetPython(int idx, handle value) {                              \
      Set(idx, CastOrTypeError<cpp_type>(value));                        \
    }                                                                    \
    void Append(handle value) {                                          \
      reflection_->Add##func_type(proto_, field_desc_,                   \
                                  CastOrTypeError<cpp_type>(value));     \
    }                                                                    \
    std::string ElementRepr(int idx) const {                             \
      return std::to_string(Get(idx));                                   \
    }                                                                    \
  }

NUMERIC_FIELD_REFLECTION_SPECIALIZATION(Int32, int32_t);
NUMERIC_FIELD_REFLECTION_SPECIALIZATION(Int64, int64_t);
NUMERIC_FIELD_REFLECTION_SPECIALIZATION(UInt32, uint32_t);
NUMERIC_FIELD_REFLECTION_SPECIALIZATION(UInt64, uint64_t);
NUMERIC_FIELD_REFLECTION_SPECIALIZATION(Float, float);
NUMERIC_FIELD_REFLECTION_SPECIALIZATION(Double, double);
NUMERIC_FIELD_REFLECTION_SPECIALIZATION(Bool, bool);
#undef NUMERIC_FIELD_REFLECTION_SPECIALIZATION

// Specialization for strings.
template <>
class ProtoFieldContainer<std::string> : public ProtoFieldContainerBase {
 public:
  using ProtoFieldContainerBase::ProtoFieldContainerBase;
  const std::string& Get(int idx) const {
    if (field_desc_->is_repeated()) {
      return reflection_->GetRepeatedStringReference(*proto_, field_desc_,
                                                     CheckIndex(idx), &scratch);
    } else {
      return reflection_->GetStringReference(*proto_, field_desc_, &scratch);
    }
  }
  object GetPython(int idx) const {
    // Convert the given value to a python string or bytes object.
    // If byte fields are returned as standard strings, pybind11 will attempt
    // to decode them as utf-8. However, some byte sequences are illegal in
    // utf-8, and will result in an error. To allow any arbitrary byte sequence
    // for a bytes field, we have to convert it to a python bytes type.
    if (field_desc_->type() == ::google::protobuf::FieldDescriptor::TYPE_BYTES)
      return bytes(Get(idx));
    else
      return str(Get(idx));
  }
  void Set(int idx, std::string value) {
    if (field_desc_->is_repeated()) {
      reflection_->SetRepeatedString(proto_, field_desc_, CheckIndex(idx),
                                     std::move(value));
    } else {
      reflection_->SetString(proto_, field_desc_, std::move(value));
    }
  }
  void SetPython(int idx, handle value) {
    Set(idx, CastOrTypeError<std::string>(value));
  }
  void Append(handle value) {
    reflection_->AddString(proto_, field_desc_,
                           CastOrTypeError<std::string>(value));
  }
  std::string ElementRepr(int idx) const {
    if (field_desc_->type() == ::google::protobuf::FieldDescriptor::TYPE_BYTES)
      return "<Binary String>";
    else
      return "'" + Get(idx) + "'";
  }

 private:
  mutable std::string scratch;
};

// Specialization for messages.
template <>
class ProtoFieldContainer<::google::protobuf::Message> : public ProtoFieldContainerBase {
 public:
  using ProtoFieldContainerBase::ProtoFieldContainerBase;
  ::google::protobuf::Message* Get(int idx) const {
    ::google::protobuf::Message* message;
    if (field_desc_->is_repeated()) {
      message = reflection_->MutableRepeatedMessage(proto_, field_desc_,
                                                    CheckIndex(idx));
    } else {
      message = reflection_->MutableMessage(proto_, field_desc_);
    }
    return message;
  }
  object GetPython(int idx) const {
    return this->CastAndKeepAlive(Get(idx), return_value_policy::reference);
  }
  void Set(int idx, ::google::protobuf::Message* value) {
    if (value->GetTypeName() != field_desc_->message_type()->full_name())
      throw type_error("Cannot set field from invalid type.");
    Get(idx)->CopyFrom(*value);
  }
  void SetPython(int idx, handle value) {
    // Value could be a native or wrapped C++ proto.
    CheckValueType(value);
    Get(idx)->ParseFromString(PyProtoSerializeToString(value));
  }
  void Append(handle value) {
    CheckValueType(value);
    reflection_->AddAllocatedMessage(
        proto_, field_desc_,
        PyProtoAllocateAndCopyMessage<::google::protobuf::Message>(value).release());
  }
  ::google::protobuf::Message* Add(kwargs kwargs_in = kwargs()) {
    // Use a unique_ptr because it will automatically free memory if
    // ProtoInitFields throws an exception.
    std::unique_ptr<::google::protobuf::Message> new_msg = std::unique_ptr<::google::protobuf::Message>(
        reflection_
            ->GetMutableRepeatedFieldRef<::google::protobuf::Message>(proto_, field_desc_)
            .NewMessage());
    ProtoInitFields(new_msg.get(), kwargs_in);
    // Transfer ownership of the new message to the proto repeated field.
    auto new_msg_raw_ptr = new_msg.release();
    reflection_->AddAllocatedMessage(proto_, field_desc_, new_msg_raw_ptr);
    return new_msg_raw_ptr;
  }
  std::string ElementRepr(int idx) const {
    return Get(idx)->ShortDebugString();
  }

 private:
  void CheckValueType(handle value) {
    if (!PyProtoCheckType(value, field_desc_->message_type()->full_name()))
      throw type_error("Cannot set field from invalid type.");
  }
};

// Specialization for enums.
template <>
class ProtoFieldContainer<GenericEnum> : public ProtoFieldContainerBase {
 public:
  using ProtoFieldContainerBase::ProtoFieldContainerBase;
  const ::google::protobuf::EnumValueDescriptor* GetDesc(int idx) const {
    if (field_desc_->is_repeated()) {
      return reflection_->GetRepeatedEnum(*proto_, field_desc_,
                                          CheckIndex(idx));
    } else {
      return reflection_->GetEnum(*proto_, field_desc_);
    }
  }
  int Get(int idx) const { return GetDesc(idx)->number(); }
  object GetPython(int idx) const { return cast(Get(idx)); }
  void Set(int idx, int value) {
    if (field_desc_->is_repeated()) {
      reflection_->SetRepeatedEnumValue(proto_, field_desc_, CheckIndex(idx),
                                        value);
    } else {
      reflection_->SetEnumValue(proto_, field_desc_, value);
    }
  }
  void SetPython(int idx, handle value) {
    Set(idx, CastOrTypeError<int>(value));
  }
  void Append(handle value) {
    reflection_->AddEnumValue(proto_, field_desc_, CastOrTypeError<int>(value));
  }
  std::string ElementRepr(int idx) const { return GetDesc(idx)->name(); }
};

// A container for a repeated field.
template <typename T>
class RepeatedFieldContainer : public ProtoFieldContainer<T> {
 public:
  using ProtoFieldContainer<T>::ProtoFieldContainer;
  object GetPythonContainer() const {
    return this->CastAndKeepAlive(this, return_value_policy::copy);
  }
  void Extend(handle src) {
    if (!isinstance<sequence>(src))
      throw std::invalid_argument("Extend: Passed value is not a sequence.");
    auto values = reinterpret_borrow<sequence>(src);
    for (auto value : values) this->Append(value);
  }
  void Insert(int idx, handle value) {
    idx = this->ClampIndex(idx);
    // Append a new element to the end.
    this->Append(value);
    // Slide all existing values up one index.
    for (int dst = this->Size() - 1; dst > idx; --dst)
      SwapElements(dst, dst - 1);
  }
  void Delete(int idx) {
    idx = this->CheckIndex(idx);
    // Slide all existing values down one index.
    for (int dst = idx; dst < this->Size() - 1; ++dst)
      SwapElements(dst, dst + 1);
    // Remove the last value
    this->reflection_->RemoveLast(this->proto_, this->field_desc_);
  }
  object GetItem(int index) { return this->GetPython(index); }
  void DelItem(int index) { Delete(index); }
  void SetItem(int index, handle value) { this->SetPython(index, value); }
  object GetSlice(slice slice) {
    size_t start, stop, step, slice_length;
    if (!slice.compute(this->Size(), &start, &stop, &step, &slice_length))
      throw error_already_set();
    list seq;
    for (size_t i = 0; i < slice_length; ++i) {
      seq.append(this->GetPython(start));
      start += step;
    }
    return seq;
  }
  void SetSlice(slice slice, handle values) {
    size_t start, stop, step, slice_length;
    if (!slice.compute(this->Size(), &start, &stop, &step, &slice_length))
      throw error_already_set();
    for (size_t i = 0; i < slice_length; ++i) {
      this->SetPython(start, values[int_(i)]);
      start += step;
    }
  }
  void DelSlice(slice slice) {
    size_t start, stop, step, slice_length, field_length = this->Size();
    if (!slice.compute(field_length, &start, &stop, &step, &slice_length))
      throw error_already_set();
    if (slice_length == field_length) {
      this->Clear();  // More efficient than removing all elements one-by-one.
    } else {
      for (size_t i = 0; i < slice_length; ++i) {
        stop -= step;
        Delete(stop);
      }
    }
  }
  std::string Repr() const {
    if (this->Size() == 0) return "[]";
    std::string repr = "[";
    for (int i = 0; i < this->Size(); ++i) repr += this->ElementRepr(i) + ", ";
    repr.pop_back();
    repr.back() = ']';
    return repr;
  }

 protected:
  void SwapElements(int i1, int i2) {
    this->reflection_->SwapElements(this->proto_, this->field_desc_, i1, i2);
  }
};

// Struct which can be used with DispatchFieldDescriptor to find (or add)
// the map pair (key, value) message with the given key.
template <typename KeyT>
struct FindMapPair {
  static ::google::protobuf::Message* HandleField(const ::google::protobuf::FieldDescriptor* key_desc,
                                      ::google::protobuf::Message* proto,
                                      const ::google::protobuf::FieldDescriptor* map_desc,
                                      handle key, bool add_key = true) {
    // When using the proto reflection API, maps are represented as repeated
    // message fields (messages with 2 elements: 'key' and 'value'). If protocol
    // buffers guarrantee a consistent (and useful) ordering of elements,
    // it should be possible to do this search in O(log(n)) time. However, that
    // requires more knowledge of protobuf internals than I have, so for now
    // assume a random ordering of elements, in which case a O(n) search is
    // the best you can do.
    RepeatedFieldContainer<::google::protobuf::Message> map_field(proto, map_desc);
    for (int i = 0; i < map_field.Size(); ++i) {
      ::google::protobuf::Message* kv_pair = map_field.Get(i);
      if (ProtoFieldContainer<KeyT>(kv_pair, key_desc).GetPython(-1).equal(key))
        return kv_pair;
    }
    // Key not found
    if (!add_key) return nullptr;
    ::google::protobuf::Message* new_kv_pair = map_field.Add();
    ProtoFieldContainer<KeyT>(new_kv_pair, key_desc).SetPython(-1, key);
    return new_kv_pair;
  }
};

// Struct which can be used with DispatchFieldDescriptor to get the value of
// the map key.
template <typename KeyType>
struct GetMapKey {
  static object HandleField(const ::google::protobuf::FieldDescriptor* key_desc,
                            ::google::protobuf::Message* kv_pair, ::google::protobuf::Message* parent) {
    return ProtoFieldContainer<KeyType>(kv_pair, key_desc, parent)
        .GetPython(-1);
  }
};

// Struct which can be used with DispatchFieldDescriptor to get the string
// representation of the map key.
template <typename KeyType>
struct GetMapKeyRepr {
  static std::string HandleField(const ::google::protobuf::FieldDescriptor* key_desc,
                                 ::google::protobuf::Message* kv_pair) {
    return ProtoFieldContainer<KeyType>(kv_pair, key_desc).ElementRepr(-1);
  }
};

// Container for a map field.
template <typename MappedType>
class MapFieldContainer : public RepeatedFieldContainer<::google::protobuf::Message> {
 public:
  MapFieldContainer(::google::protobuf::Message* proto,
                    const ::google::protobuf::FieldDescriptor* map_desc)
      : RepeatedFieldContainer<::google::protobuf::Message>(proto, map_desc),
        key_desc_(map_desc->message_type()->FindFieldByName("key")),
        value_desc_(map_desc->message_type()->FindFieldByName("value")) {}
  using RepeatedFieldContainer<::google::protobuf::Message>::RepeatedFieldContainer;
  object GetPythonContainer() const {
    return this->CastAndKeepAlive(this, return_value_policy::copy);
  }
  // GetItem automatically inserts missing keys, which matches native
  // python protos, not dicts (see http://go/pythonprotobuf#undefined).
  object GetItem(handle key) const {
    return GetValueContainer(key).GetPython(-1);
  }
  void SetItem(handle key, handle value) {
    GetValueContainer(key).SetPython(-1, value);
  }
  void UpdateFromDict(dict values) {
    for (auto& item : values) SetItem(item.first, item.second);
  }
  void UpdateFromKWArgs(kwargs values) { UpdateFromDict(values); }
  void UpdateFromHandle(handle values) {
    if (!isinstance<dict>(values))
      throw std::invalid_argument("Update: Passed value is not a dictionary.");
    UpdateFromDict(reinterpret_borrow<dict>(values));
  }
  bool Contains(handle key) const {
    return DispatchFieldDescriptor<FindMapPair>(key_desc_, proto_, field_desc_,
                                                key, false) != nullptr;
  }
  std::string Repr() const {
    if (Size() == 0) return "{}";
    std::string repr = "{";
    for (int i = 0; i < Size(); ++i) {
      ::google::protobuf::Message* kv_pair = Get(i);
      repr += DispatchFieldDescriptor<GetMapKeyRepr>(key_desc_, kv_pair) +
              ": " +
              ProtoFieldContainer<MappedType>(kv_pair, value_desc_)
                  .ElementRepr(-1) +
              ", ";
    }
    repr.pop_back();
    repr.back() = '}';
    return repr;
  }
  std::function<std::unique_ptr<::google::protobuf::Message>(kwargs)>
  GetEntryClassFactory() {
    return [descriptor = field_desc_->message_type()](kwargs kwargs_in) {
      return PyProtoAllocateMessage(descriptor, kwargs_in);
    };
  }

  // Iterator class for maps.
  class Iterator {
   public:
    // First argument is the container, second is a member function to extract
    // the appropriate value from a key-value pair message (ie, this function
    // should return either the key, the value, or a (key, value) tuple).
    Iterator(MapFieldContainer* container,
             object (MapFieldContainer::*value_helper)(::google::protobuf::Message*))
        : container_(container), value_helper_(value_helper) {}

    auto* iter() { return this; }
    object next() {
      if (idx_ >= container_->Size()) throw stop_iteration();
      return (container_->*value_helper_)(container_->Get(idx_++));
    }

   private:
    MapFieldContainer* container_;
    object (MapFieldContainer::*value_helper_)(::google::protobuf::Message*);
    int idx_ = 0;
  };

  Iterator KeyIterator() { return Iterator(this, &MapFieldContainer::GetKey); }
  Iterator ValueIterator() {
    return Iterator(this, &MapFieldContainer::GetValue);
  }
  Iterator ItemIterator() {
    return Iterator(this, &MapFieldContainer::GetTuple);
  }

 protected:
  // Note that the helper functions bellow all pass the message which contains
  // the key-value pair to the ProtoFieldContainer as the 3rd argument.

  // Get the ProtoFieldContainer for the value corresponding to the given key.
  ProtoFieldContainer<MappedType> GetValueContainer(handle key) const {
    ::google::protobuf::Message* kv_pair = DispatchFieldDescriptor<FindMapPair>(
        key_desc_, proto_, field_desc_, key);
    return ProtoFieldContainer<MappedType>(kv_pair, value_desc_, proto_);
  }

  // Get the key out of the given key-value message.
  object GetKey(::google::protobuf::Message* kv_pair) {
    return DispatchFieldDescriptor<GetMapKey>(key_desc_, kv_pair, proto_);
  }

  // Get the value out of the given key-value message.
  object GetValue(::google::protobuf::Message* kv_pair) {
    return ProtoFieldContainer<MappedType>(kv_pair, value_desc_, proto_)
        .GetPython(-1);
  }

  // Get the given key-value pair as a message.
  object GetTuple(::google::protobuf::Message* kv_pair) {
    return make_tuple(GetKey(kv_pair), GetValue(kv_pair));
  }

  const ::google::protobuf::FieldDescriptor* key_desc_;
  const ::google::protobuf::FieldDescriptor* value_desc_;
};

const ::google::protobuf::FieldDescriptor* GetFieldDescriptor(
    ::google::protobuf::Message* message, absl::string_view name,
    PyObject* error_type = PyExc_AttributeError) {
  auto* field_desc =
  message->GetDescriptor()->FindFieldByName(std::string(name));

  if (!field_desc) {
    std::string error_str =
        "'" + message->GetTypeName() + "' object has no attribute '";
    error_str.append(std::string(name));
    error_str.append("'");
    PyErr_SetString(error_type, error_str.c_str());
    throw error_already_set();
  }
  return field_desc;
}

// Struct used with DispatchFieldDescriptor to get the value of a field.
template <typename ValueType>
struct TemplatedProtoGetField {
  static object HandleField(const ::google::protobuf::FieldDescriptor* field_desc,
                            ::google::protobuf::Message* proto) {
    if (field_desc->is_map()) {
      return MapFieldContainer<ValueType>(proto, field_desc)
          .GetPythonContainer();
    } else if (field_desc->is_repeated()) {
      return RepeatedFieldContainer<ValueType>(proto, field_desc)
          .GetPythonContainer();
    } else {  // Singular field.
      return ProtoFieldContainer<ValueType>(proto, field_desc).GetPython(-1);
    }
  }
};

// Struct used with DispatchFieldDescriptor to set the value of a field.
template <typename ValueType>
struct TemplatedProtoSetField {
  static void HandleField(const ::google::protobuf::FieldDescriptor* field_desc,
                          ::google::protobuf::Message* proto, handle value) {
    if (field_desc->is_map()) {
      MapFieldContainer<ValueType> map_field(proto, field_desc);
      map_field.Clear();
      map_field.UpdateFromHandle(value);
    } else if (field_desc->is_repeated()) {
      RepeatedFieldContainer<ValueType> repeated_field(proto, field_desc);
      repeated_field.Clear();
      repeated_field.Extend(value);
    } else {  // Singular field.
      ProtoFieldContainer<ValueType>(proto, field_desc).SetPython(-1, value);
    }
  }
};

// Wrapper around ::google::protobuf::Message::FindInitializationErrors.
std::vector<std::string> MessageFindInitializationErrors(
    ::google::protobuf::Message* message) {
  std::vector<std::string> errors;
  message->FindInitializationErrors(&errors);
  return errors;
}

// Wrapper around ::google::protobuf::Message::ListFields.
std::vector<tuple> MessageListFields(::google::protobuf::Message* message) {
  std::vector<const ::google::protobuf::FieldDescriptor*> fields;
  message->GetReflection()->ListFields(*message, &fields);
  std::vector<tuple> result;
  result.reserve(fields.size());
  for (auto* field_desc : fields) {
    result.push_back(
        make_tuple(cast(field_desc, return_value_policy::reference),
                   ProtoGetField(message, field_desc)));
  }
  return result;
}

// Wrapper around ::google::protobuf::Message::HasField.
bool MessageHasField(::google::protobuf::Message* message, absl::string_view field_name) {
  auto* oneof_desc = message->GetDescriptor()->FindOneofByName(std::string(field_name));
  if (oneof_desc) {
    return message->GetReflection()->HasOneof(*message, oneof_desc);
  }
  auto* field_desc = GetFieldDescriptor(message, field_name, PyExc_ValueError);
  return message->GetReflection()->HasField(*message, field_desc);
}

void MessageClearField(::google::protobuf::Message* message, absl::string_view field_name) {
  auto* oneof_desc = message->GetDescriptor()->FindOneofByName(std::string(field_name));
  if (oneof_desc) {
    message->GetReflection()->ClearOneof(message, oneof_desc);
    return;
  }

  auto* field_desc = GetFieldDescriptor(message, field_name, PyExc_ValueError);
  message->GetReflection()->ClearField(message, field_desc);
}

const std::string* MessageWhichOneof(::google::protobuf::Message* message,
                                     absl::string_view oneof_group) {
  auto* oneof_desc = message->GetDescriptor()->FindOneofByName(std::string(oneof_group));
  if (!oneof_desc) {
    std::string error_str = "Requested oneof does not exist: ";
    error_str.append(std::string(oneof_group));
    throw std::invalid_argument(error_str);
  }

  auto* field_desc =
      message->GetReflection()->GetOneofFieldDescriptor(*message, oneof_desc);
  if (!field_desc) return nullptr;
  return &field_desc->name();
}

// Wrapper around ::google::protobuf::Message::SerializeAsString.
// The only valid kwarg is "deterministic", which should be a boolean and, if
// true, causes maps to be serialized with deterministic order. Keyword
// arguments are used here because the native python implementation does not
// allow this to be passed as a positional argument, and we want to match that.
bytes MessageSerializeAsString(::google::protobuf::Message* msg, kwargs kwargs_in,
                               bool partial) {
  static constexpr char kwargs_key[] = "deterministic";
  std::string result;
  bool deterministic = false;
  if (!kwargs_in.empty()) {
    if (kwargs_in.size() == 1 && kwargs_in.contains(kwargs_key)) {
      deterministic = bool_(kwargs_in[kwargs_key]);
    } else {
      throw std::invalid_argument(
          "Invalid kwargs; the only valid key is 'deterministic'");
    }
  }
  if (deterministic) {
    ::google::protobuf::io::StringOutputStream string_stream(&result);
    ::google::protobuf::io::CodedOutputStream coded_stream(&string_stream);
    coded_stream.SetSerializationDeterministic(true);
    if (partial) {
      msg->SerializePartialToCodedStream(&coded_stream);
    } else {
      msg->SerializeToCodedStream(&coded_stream);
    }
  } else {
    if (partial) {
      result = msg->SerializePartialAsString();
    } else {
      result = msg->SerializeAsString();
    }
  }
  return bytes(result);
}

// Wrapper to generate the python message Descriptor.fields property.
list MessageFields(const ::google::protobuf::Descriptor* descriptor) {
  list result;
  for (int i = 0; i < descriptor->field_count(); ++i)
    result.append(cast(descriptor->field(i), return_value_policy::reference));
  return result;
}

// Wrapper to generate the python message Descriptor.fields_by_name property.
dict MessageFieldsByName(const ::google::protobuf::Descriptor* descriptor) {
  dict result;
  for (int i = 0; i < descriptor->field_count(); ++i) {
    auto* field_desc = descriptor->field(i);
    result[cast(field_desc->name())] =
        cast(field_desc, return_value_policy::reference);
  }
  return result;
}

// Wrapper to generate the python EnumDescriptor.values_by_number property.
dict EnumValuesByNumber(const ::google::protobuf::EnumDescriptor* enum_descriptor) {
  dict result;
  for (int i = 0; i < enum_descriptor->value_count(); ++i) {
    auto* value_desc = enum_descriptor->value(i);
    result[cast(value_desc->number())] =
        cast(value_desc, return_value_policy::reference);
  }
  return result;
}

// Wrapper to generate the python EnumDescriptor.values_by_name property.
dict EnumValuesByName(const ::google::protobuf::EnumDescriptor* enum_descriptor) {
  dict result;
  for (int i = 0; i < enum_descriptor->value_count(); ++i) {
    auto* value_desc = enum_descriptor->value(i);
    result[cast(value_desc->name())] =
        cast(value_desc, return_value_policy::reference);
  }
  return result;
}

// Class to add python bindings to a RepeatedFieldContainer.
// This corresponds to the repeated field API:
// https://developers.google.com/protocol-buffers/docs/reference/python-generated#repeated-fields
// https://developers.google.com/protocol-buffers/docs/reference/python-generated#repeated-message-fields
template <typename T>
class RepeatedFieldBindings : public class_<RepeatedFieldContainer<T>> {
 public:
  RepeatedFieldBindings(handle scope, const std::string& name)
      : class_<RepeatedFieldContainer<T>>(scope, name.c_str()) {
    // Repeated message fields support `add` but not `__setitem__`.
    if (std::is_same<T, ::google::protobuf::Message>::value) {
      this->def("add", &RepeatedFieldContainer<T>::Add,
                return_value_policy::reference_internal);
    } else {
      this->def("__setitem__", &RepeatedFieldContainer<T>::SetItem);
      this->def("__setitem__", &RepeatedFieldContainer<T>::SetSlice);
    }
    this->def("__repr__", &RepeatedFieldContainer<T>::Repr);
    this->def("__len__", &RepeatedFieldContainer<T>::Size);
    this->def("__getitem__", &RepeatedFieldContainer<T>::GetItem);
    this->def("__getitem__", &RepeatedFieldContainer<T>::GetSlice);
    this->def("__delitem__", &RepeatedFieldContainer<T>::DelItem);
    this->def("__delitem__", &RepeatedFieldContainer<T>::DelSlice);
    this->def("MergeFrom", &RepeatedFieldContainer<T>::Extend);
    this->def("extend", &RepeatedFieldContainer<T>::Extend);
    this->def("append", &RepeatedFieldContainer<T>::Append);
    this->def("insert", &RepeatedFieldContainer<T>::Insert);
  }
};

// Class to add python bindings to a MapFieldContainer.
// This corresponds to the map field API:
// https://developers.google.com/protocol-buffers/docs/reference/python-generated#map-fields
template <typename T>
class MapFieldBindings : public class_<MapFieldContainer<T>> {
 public:
  MapFieldBindings(handle scope, const std::string& name)
      : class_<MapFieldContainer<T>>(scope, name.c_str()) {
    // Mapped message fields don't support item assignment.
    if (std::is_same<T, ::google::protobuf::Message>::value) {
      this->def("__setitem__", [](void*, int, handle) {
        throw value_error("Cannot assign to message in a map field.");
      });
    } else {
      this->def("__setitem__", &MapFieldContainer<T>::SetItem);
    }
    this->def("__repr__", &MapFieldContainer<T>::Repr);
    this->def("__len__", &MapFieldContainer<T>::Size);
    this->def("__contains__", &MapFieldContainer<T>::Contains);
    this->def("__getitem__", &MapFieldContainer<T>::GetItem);
    this->def("__iter__", &MapFieldContainer<T>::KeyIterator,
              keep_alive<0, 1>());
    this->def("keys", &MapFieldContainer<T>::KeyIterator, keep_alive<0, 1>());
    this->def("values", &MapFieldContainer<T>::ValueIterator,
              keep_alive<0, 1>());
    this->def("items", &MapFieldContainer<T>::ItemIterator, keep_alive<0, 1>());
    this->def("update", &MapFieldContainer<T>::UpdateFromDict);
    this->def("update", &MapFieldContainer<T>::UpdateFromKWArgs);
    this->def("clear", &MapFieldContainer<T>::Clear);
    this->def("GetEntryClass", &MapFieldContainer<T>::GetEntryClassFactory,
              "Returns a factory function which can be called with keyword "
              "Arguments to create an instance of the map entry class (ie, "
              "a message with `key` and `value` fields). Used by text_format.");
  }
};

// Class to add python bindings to a map field iterator.
template <typename T>
class MapFieldIteratorBindings
    : public class_<typename MapFieldContainer<T>::Iterator> {
 public:
  MapFieldIteratorBindings(handle scope, const std::string& name)
      : class_<typename MapFieldContainer<T>::Iterator>(
            scope, (name + "Iterator").c_str()) {
    this->def("__iter__", &MapFieldContainer<T>::Iterator::iter);
    this->def("__next__", &MapFieldContainer<T>::Iterator::next);
  }
};

// Function to instantiate the given Bindings class with each of the types
// supported by protobuffers.
template <template <typename> class Bindings>
void BindEachFieldType(module& module, const std::string& name) {
  Bindings<int32_t>(module, name + "Int32");
  Bindings<int64_t>(module, name + "Int64");
  Bindings<uint32_t>(module, name + "UInt32");
  Bindings<uint64_t>(module, name + "UInt64");
  Bindings<float>(module, name + "Float");
  Bindings<double>(module, name + "Double");
  Bindings<bool>(module, name + "Bool");
  Bindings<std::string>(module, name + "String");
  Bindings<::google::protobuf::Message>(module, name + "Message");
  Bindings<GenericEnum>(module, name + "Enum");
}

// Define a property in the given class_ with the given name which is constant
// for the life of an object instance. The generator function will be invoked
// the first time the property is accessed. The result of this function will
// be cached and returned for all future accesses, without invoking the
// generator function again. dynamic_attr() must have been passed to the class_
// constructor or you will get "AttributeError: can't set attribute".
// The given policy is applied to the return value of the generator function.
template <typename Class, typename Return, typename... Args>
void DefConstantProperty(
    class_<detail::intrinsic_t<Class>>* pyclass, const std::string& name,
    std::function<Return(Class, Args...)> generator,
    return_value_policy policy = return_value_policy::automatic_reference) {
  auto wrapper = [generator, name, policy](handle pyinst, Args... args) {
    std::string cache_name = "_cache_" + name;
    if (!hasattr(pyinst, cache_name.c_str())) {
      // Invoke the generator and cache the result.
      Return result =
          generator(cast<Class>(pyinst), std::forward<Args>(args)...);
      setattr(
          pyinst, cache_name.c_str(),
          detail::make_caster<object>::cast(std::move(result), policy, pyinst));
    }
    return getattr(pyinst, cache_name.c_str());
  };
  pyclass->def_property_readonly(name.c_str(), wrapper);
}

// Same as above, but for a function pointer rather than a std::function.
template <typename Class, typename Return, typename... Args>
void DefConstantProperty(
    class_<detail::intrinsic_t<Class>>* pyclass, const std::string& name,
    Return (*generator)(Class, Args...),
    return_value_policy policy = return_value_policy::automatic_reference) {
  DefConstantProperty(pyclass, name,
                      std::function<Return(Class, Args...)>(generator), policy);
}

}  // namespace

bool PyProtoFullName(handle py_proto, std::string* name) {
  if (hasattr(py_proto, "DESCRIPTOR")) {
    auto descriptor = py_proto.attr("DESCRIPTOR");
    if (hasattr(descriptor, "full_name")) {
      if (name) *name = cast<std::string>(descriptor.attr("full_name"));
      return true;
    }
  }
  return false;
}

bool PyProtoCheckType(handle py_proto, const std::string& expected_type) {
  std::string name;
  if (PyProtoFullName(py_proto, &name)) return name == expected_type;
  return false;
}

void PyProtoCheckTypeOrThrow(handle py_proto,
                             const std::string& expected_type) {
  std::string name;
  if (!PyProtoFullName(py_proto, &name)) {
    auto builtins = module::import(PYBIND11_BUILTINS_MODULE);
    std::string type_str =
        str(builtins.attr("repr")(builtins.attr("type")(py_proto)));
    throw type_error("Expected a proto, got a " + type_str + ".");
  } else if (name != expected_type) {
    throw type_error("Passed proto is the wrong type. Expected " +
                     expected_type + " but got " + name + ".");
  }
}

std::string PyProtoSerializeToString(handle py_proto) {
  if (hasattr(py_proto, "SerializeToString"))
    return cast<std::string>(py_proto.attr("SerializeToString")());
  throw std::invalid_argument("Passed python object is not a proto.");
}

const ::google::protobuf::Descriptor* PyProtoGetDescriptor(handle py_proto) {
  detail::make_caster<::google::protobuf::Message> caster;
  if (caster.load(py_proto, true)) {
    // Native C++ proto, so we can get the descriptor directly.
    return detail::cast_op<::google::protobuf::Message*>(caster)->GetDescriptor();
  }

  // Look the descriptor based on the proto's type name.
  std::string full_type_name;
  if (isinstance<bytes>(py_proto)) {
    full_type_name = py_proto.cast<std::string>();
  } else if (isinstance<str>(py_proto)) {
    full_type_name = str(py_proto);
  } else if (!PyProtoFullName(py_proto, &full_type_name)) {
    throw std::invalid_argument("Could not get the name of the proto.");
  }
  const ::google::protobuf::Descriptor* descriptor =
      ::google::protobuf::DescriptorPool::generated_pool()->FindMessageTypeByName(
          full_type_name);
  if (!descriptor)
    throw std::runtime_error("Proto Descriptor not found: " + full_type_name);
  return descriptor;
}

template <>
std::unique_ptr<::google::protobuf::Message> PyProtoAllocateMessage(handle py_proto,
                                                        kwargs kwargs_in) {
  return PyProtoAllocateMessage(PyProtoGetDescriptor(py_proto), kwargs_in);
}

std::unique_ptr<::google::protobuf::Message> PyProtoAllocateMessage(
    const ::google::protobuf::Descriptor* descriptor, kwargs kwargs_in) {
  const ::google::protobuf::Message* prototype =
      ::google::protobuf::MessageFactory::generated_factory()->GetPrototype(descriptor);
  if (!prototype) {
    throw std::runtime_error(
        "Not able to generate prototype for descriptor of: " +
        descriptor->full_name());
  }
  auto message = std::unique_ptr<::google::protobuf::Message>(prototype->New());
  ProtoInitFields(message.get(), kwargs_in);
  return message;
}

bool AnyPackFromPyProto(handle py_proto, ::google::protobuf::Any* any_proto) {
  std::string name;
  if (!PyProtoFullName(py_proto, &name)) return false;
  any_proto->set_type_url("type.googleapis.com/" + name);
  any_proto->set_value(PyProtoSerializeToString(py_proto));
  return true;
}

bool AnyUnpackToPyProto(const ::google::protobuf::Any& any_proto,
                        handle py_proto) {
  // Check that py_proto is a proto message of the same type that is stored
  // in the any_proto.
  std::string any_type, proto_type;
  if (!(PyProtoFullName(py_proto, &proto_type) &&
        ::google::protobuf::Any::ParseAnyTypeUrl(
            std::string(any_proto.type_url()), &any_type) &&
        proto_type == any_type))
    return false;
  // Unpack. The serialized string is not copied if py_proto is a wrapped C
  // proto, and copied once if py_proto is a native python proto.
  detail::type_caster_base<::google::protobuf::Message> caster;
  if (caster.load(py_proto, false)) {
    return static_cast<::google::protobuf::Message&>(caster).ParseFromString(
         std::string(any_proto.value()));
  } else {
    bytes serialized(nullptr, any_proto.value().size());
    std::string any_string = any_proto.value();
    strncpy(PYBIND11_BYTES_AS_STRING(serialized.ptr()),
            any_string.c_str(), any_string.size());
    getattr(py_proto, "ParseFromString")(serialized);
    return true;
  }
}

object ProtoGetField(::google::protobuf::Message* message, absl::string_view name) {
  return ProtoGetField(message, GetFieldDescriptor(message, name));
}

object ProtoGetField(::google::protobuf::Message* message,
                     const ::google::protobuf::FieldDescriptor* field_desc) {
  return DispatchFieldDescriptor<TemplatedProtoGetField>(field_desc, message);
}

void ProtoSetField(::google::protobuf::Message* message, absl::string_view name,
                   handle value) {
  ProtoSetField(message, GetFieldDescriptor(message, name), value);
}

void ProtoSetField(::google::protobuf::Message* message,
                   const ::google::protobuf::FieldDescriptor* field_desc, handle value) {
  if (field_desc->is_map() || field_desc->is_repeated() ||
      field_desc->type() == ::google::protobuf::FieldDescriptor::TYPE_MESSAGE) {
    std::string error = "Assignment not allowed to field \"" +
                        field_desc->name() + "\" in protocol message object.";
    PyErr_SetString(PyExc_AttributeError, error.c_str());
    throw error_already_set();
  }
  DispatchFieldDescriptor<TemplatedProtoSetField>(field_desc, message, value);
}

void ProtoInitFields(::google::protobuf::Message* message, kwargs kwargs_in) {
  // NOTE: This uses pybind11 casters to construct strings, so it needs to
  // be wrapped by loader_life_support.
  pybind11::detail::loader_life_support life_support;

  for (auto& item : kwargs_in) {
    DispatchFieldDescriptor<TemplatedProtoSetField>(
        GetFieldDescriptor(message, cast<absl::string_view>(item.first)),
        message, item.second);
  }
}

void ProtoCopyFrom(::google::protobuf::Message* msg, handle other) {
  PyProtoCheckTypeOrThrow(other, msg->GetTypeName());
  detail::type_caster_base<::google::protobuf::Message> caster;
  if (caster.load(other, false)) {
    msg->CopyFrom(static_cast<::google::protobuf::Message&>(caster));
  } else {
    if (!msg->ParseFromString(PyProtoSerializeToString(other)))
      throw std::runtime_error("Error copying message.");
  }
}

void ProtoMergeFrom(::google::protobuf::Message* msg, handle other) {
  PyProtoCheckTypeOrThrow(other, msg->GetTypeName());
  detail::type_caster_base<::google::protobuf::Message> caster;
  if (caster.load(other, false)) {
    msg->MergeFrom(static_cast<::google::protobuf::Message&>(caster));
  } else {
    if (!msg->MergeFromString(PyProtoSerializeToString(other)))
      throw std::runtime_error("Error merging message.");
  }
}

void RegisterProtoBindings(module m) {
  // Return whether the given python object is a wrapped C proto.
  m.def("is_wrapped_c_proto_deprecated", &IsWrappedCProto, arg("src"));

  // Construct and optionally initialize a wrapped C proto.
  m.def("make_wrapped_c_proto", &PyProtoAllocateMessage<::google::protobuf::Message>,
        arg("type"),
        "Returns a wrapped C proto of the given type. The type may be passed "
        "as a string ('package_name.MessageName'), an instance of a native "
        "python proto, or an instance of a wrapped C proto. Fields may be "
        "initialized with keyword arguments, as with the native constructors. "
        "The C++ version of the proto library for your message type must be "
        "linked in for this to work.");

  // Add bindings for the descriptor class.
  class_<::google::protobuf::Descriptor> message_desc_c(m, "Descriptor", dynamic_attr());
  DefConstantProperty(&message_desc_c, "fields_by_name", &MessageFieldsByName);
  DefConstantProperty(&message_desc_c, "fields", &MessageFields);
  message_desc_c
      .def_property_readonly("full_name", &::google::protobuf::Descriptor::full_name)
      .def_property_readonly("name", &::google::protobuf::Descriptor::name)
      .def_property_readonly("has_options",
                             [](::google::protobuf::Descriptor*) { return true; })
      .def(
          "GetOptions",
          [](::google::protobuf::Descriptor* descriptor) -> const ::google::protobuf::Message* {
            return &descriptor->options();
          },
          return_value_policy::reference);

  // Add bindings for the enum descriptor class.
  class_<::google::protobuf::EnumDescriptor> enum_desc_c(m, "EnumDescriptor",
                                             dynamic_attr());
  DefConstantProperty(&enum_desc_c, "values_by_number", &EnumValuesByNumber);
  DefConstantProperty(&enum_desc_c, "values_by_name", &EnumValuesByName);
  enum_desc_c.def_property_readonly("name", &::google::protobuf::EnumDescriptor::name);

  // Add bindings for the enum value descriptor class.
  class_<::google::protobuf::EnumValueDescriptor>(m, "EnumValueDescriptor")
      .def_property_readonly("name", &::google::protobuf::EnumValueDescriptor::name)
      .def_property_readonly("number", &::google::protobuf::EnumValueDescriptor::number);

  // Add bindings for the field descriptor class.
  class_<::google::protobuf::FieldDescriptor> field_desc_c(m, "FieldDescriptor");
  field_desc_c.def_property_readonly("name", &::google::protobuf::FieldDescriptor::name)
      .def_property_readonly("type", &::google::protobuf::FieldDescriptor::type)
      .def_property_readonly("cpp_type", &::google::protobuf::FieldDescriptor::cpp_type)
      .def_property_readonly("containing_type",
                             &::google::protobuf::FieldDescriptor::containing_type,
                             return_value_policy::reference)
      .def_property_readonly("message_type",
                             &::google::protobuf::FieldDescriptor::message_type,
                             return_value_policy::reference)
      .def_property_readonly("enum_type", &::google::protobuf::FieldDescriptor::enum_type,
                             return_value_policy::reference)
      .def_property_readonly("is_extension",
                             &::google::protobuf::FieldDescriptor::is_extension)
      .def_property_readonly("label", &::google::protobuf::FieldDescriptor::label)
      // Oneof fields are not currently supported.
      .def_property_readonly("containing_oneof", [](void*) { return false; });

  // Add Type enum values.
  enum_<::google::protobuf::FieldDescriptor::Type>(field_desc_c, "Type")
      .value("TYPE_DOUBLE", ::google::protobuf::FieldDescriptor::TYPE_DOUBLE)
      .value("TYPE_FLOAT", ::google::protobuf::FieldDescriptor::TYPE_FLOAT)
      .value("TYPE_INT64", ::google::protobuf::FieldDescriptor::TYPE_INT64)
      .value("TYPE_UINT64", ::google::protobuf::FieldDescriptor::TYPE_UINT64)
      .value("TYPE_INT32", ::google::protobuf::FieldDescriptor::TYPE_INT32)
      .value("TYPE_FIXED64", ::google::protobuf::FieldDescriptor::TYPE_FIXED64)
      .value("TYPE_FIXED32", ::google::protobuf::FieldDescriptor::TYPE_FIXED32)
      .value("TYPE_BOOL", ::google::protobuf::FieldDescriptor::TYPE_BOOL)
      .value("TYPE_STRING", ::google::protobuf::FieldDescriptor::TYPE_STRING)
      .value("TYPE_GROUP", ::google::protobuf::FieldDescriptor::TYPE_GROUP)
      .value("TYPE_MESSAGE", ::google::protobuf::FieldDescriptor::TYPE_MESSAGE)
      .value("TYPE_BYTES", ::google::protobuf::FieldDescriptor::TYPE_BYTES)
      .value("TYPE_UINT32", ::google::protobuf::FieldDescriptor::TYPE_UINT32)
      .value("TYPE_ENUM", ::google::protobuf::FieldDescriptor::TYPE_ENUM)
      .value("TYPE_SFIXED32", ::google::protobuf::FieldDescriptor::TYPE_SFIXED32)
      .value("TYPE_SFIXED64", ::google::protobuf::FieldDescriptor::TYPE_SFIXED64)
      .value("TYPE_SINT32", ::google::protobuf::FieldDescriptor::TYPE_SINT32)
      .value("TYPE_SINT64", ::google::protobuf::FieldDescriptor::TYPE_SINT64)
      .export_values();

  // Add C++ Type enum values.
  enum_<::google::protobuf::FieldDescriptor::CppType>(field_desc_c, "CppType")
      .value("CPPTYPE_INT32", ::google::protobuf::FieldDescriptor::CPPTYPE_INT32)
      .value("CPPTYPE_INT64", ::google::protobuf::FieldDescriptor::CPPTYPE_INT64)
      .value("CPPTYPE_UINT32", ::google::protobuf::FieldDescriptor::CPPTYPE_UINT32)
      .value("CPPTYPE_UINT64", ::google::protobuf::FieldDescriptor::CPPTYPE_UINT64)
      .value("CPPTYPE_DOUBLE", ::google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE)
      .value("CPPTYPE_FLOAT", ::google::protobuf::FieldDescriptor::CPPTYPE_FLOAT)
      .value("CPPTYPE_BOOL", ::google::protobuf::FieldDescriptor::CPPTYPE_BOOL)
      .value("CPPTYPE_ENUM", ::google::protobuf::FieldDescriptor::CPPTYPE_ENUM)
      .value("CPPTYPE_STRING", ::google::protobuf::FieldDescriptor::CPPTYPE_STRING)
      .value("CPPTYPE_MESSAGE", ::google::protobuf::FieldDescriptor::CPPTYPE_MESSAGE)
      .export_values();

  // Add Label enum values.
  enum_<::google::protobuf::FieldDescriptor::Label>(field_desc_c, "Label")
      .value("LABEL_OPTIONAL", ::google::protobuf::FieldDescriptor::LABEL_OPTIONAL)
      .value("LABEL_REQUIRED", ::google::protobuf::FieldDescriptor::LABEL_REQUIRED)
      .value("LABEL_REPEATED", ::google::protobuf::FieldDescriptor::LABEL_REPEATED)
      .export_values();

  // Add bindings for the base message class.
  // These bindings use __getattr__, __setattr__ and the reflection interface
  // to access message-specific fields, so no additional bindings are needed
  // for derived message types.
  class_<::google::protobuf::Message, std::shared_ptr<::google::protobuf::Message>>(m, "ProtoMessage")
      .def_property_readonly("DESCRIPTOR", &::google::protobuf::Message::GetDescriptor,
                             return_value_policy::reference)
      .def_property_readonly(kIsWrappedCProtoAttr, [](void*) { return true; })
      .def("__repr__", &::google::protobuf::Message::DebugString)
      .def("__getattr__",
           (object(*)(::google::protobuf::Message*, absl::string_view)) & ProtoGetField)
      .def("__setattr__",
           (void (*)(::google::protobuf::Message*, absl::string_view, handle)) &
               ProtoSetField)
      .def("SerializeToString",
           [](::google::protobuf::Message* msg, kwargs kwargs_in) {
             return MessageSerializeAsString(msg, kwargs_in, false);
           })
      .def("SerializePartialToString",
           [](::google::protobuf::Message* msg, kwargs kwargs_in) {
             return MessageSerializeAsString(msg, kwargs_in, true);
           })
      .def("ParseFromString", &::google::protobuf::Message::ParseFromString, arg("data"))
      .def("MergeFromString", &::google::protobuf::Message::MergeFromString, arg("data"))
      .def("ByteSize", &::google::protobuf::Message::ByteSizeLong)
      .def("Clear", &::google::protobuf::Message::Clear)
      .def("CopyFrom", &ProtoCopyFrom)
      .def("MergeFrom", &ProtoMergeFrom)
      .def("FindInitializationErrors", &MessageFindInitializationErrors,
           "Slowly build a list of all required fields that are not set.")
      .def("ListFields", &MessageListFields)
      .def("HasField", &MessageHasField, arg("field_name"))
      .def("ClearField", &MessageClearField, arg("field_name"))
      .def("WhichOneof", &MessageWhichOneof, arg("oneof_group"),
           return_value_policy::copy)
      .def(MakePickler<::google::protobuf::Message>())
      .def("__copy__",
           [](object self) {
             return PyProtoAllocateAndCopyMessage<::google::protobuf::Message>(self);
           })
      .def(
          "__deepcopy__",
          [](object self, dict) {
            return PyProtoAllocateAndCopyMessage<::google::protobuf::Message>(self);
          },
          arg("memo"))
      .def(
          "SetInParent", [](::google::protobuf::Message*) {},
          "No-op. Provided only for compatability with text_format.");

  // Add bindings for the repeated field containers.
  BindEachFieldType<RepeatedFieldBindings>(m, "Repeated");

  // Add bindings for the map field containers.
  BindEachFieldType<MapFieldBindings>(m, "Mapped");

  // Add bindings for the map field iterators.
  BindEachFieldType<MapFieldIteratorBindings>(m, "Mapped");

  // Add bindings for the well-known-types.
  // TODO(rwgk): Consider adding support for other types/methods defined in
  // google3/net/proto2/python/internal/well_known_types.py
  ConcreteProtoMessageBindings<::google::protobuf::Any>(m)
      .def("Is",
           [](const ::google::protobuf::Any& self, handle descriptor) {
             std::string name;
             if (!::google::protobuf::Any::ParseAnyTypeUrl(
                     std::string(self.type_url()), &name))
               return false;
             return name == static_cast<std::string>(
                                str(getattr(descriptor, "full_name")));
           })
      .def("TypeName",
           [](const ::google::protobuf::Any& self) {
             std::string name;
             ::google::protobuf::Any::ParseAnyTypeUrl(
                 std::string(self.type_url()), &name);
             return name;
           })
      .def("Pack",
           [](::google::protobuf::Any* self, handle py_proto) {
             if (!AnyPackFromPyProto(py_proto, self))
               throw std::invalid_argument("Failed to pack Any proto.");
           })
      .def("Unpack", &AnyUnpackToPyProto);
}

}  // namespace google
}  // namespace pybind11
