#ifndef PYBIND11_PROTOBUF_PROTO_CASTER_IMPL_H_
#define PYBIND11_PROTOBUF_PROTO_CASTER_IMPL_H_

#include <Python.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "google/protobuf/descriptor.pb.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "pybind11_protobuf/proto_cast_util.h"

// Enables unsafe conversions; currently these are a work in progress.
#if !defined(PYBIND11_PROTOBUF_UNSAFE)
#define PYBIND11_PROTOBUF_UNSAFE 0
#endif

namespace pybind11_protobuf {

// pybind11 constructs c++ references using the following mechanism, for
// example:
//
// type_caster<T> caster;
// caster.load(handle, /*convert=*/ false);
// call(pybind11::detail::cast_op<const T&>(caster));
//
template <typename ProtoType>
struct proto_caster_load_impl {
  static_assert(
      std::is_same<ProtoType, pybind11::detail::intrinsic_t<ProtoType>>::value,
      "");

  // load converts from Python -> C++
  bool load(pybind11::handle src, bool convert) {
    // When given a none, treat it as a nullptr.
    if (src.is_none()) {
      value = nullptr;
      return true;
    }
    // NOTE: We might need to know whether the proto has extensions that
    // are python-only.

    // Attempt to use the PyProto_API to get an underlying C++ message pointer
    // from the object.
    const ::google::protobuf::Message *message =
        pybind11_protobuf::PyProtoGetCppMessagePointer(src);
    if (message && message->GetReflection() ==
                       ProtoType::default_instance().GetReflection()) {
      // If the capability were available, then we could probe PyProto_API and
      // allow c++ mutability based on the python reference count.
      value = static_cast<const ProtoType *>(message);
      return true;
    }

    // The incoming object is not a compatible fast_cpp_proto, so check whether
    // it is otherwise compatible, then serialize it and deserialize into a
    // native C++ proto type.
    if (!pybind11_protobuf::PyProtoIsCompatible(src, ProtoType::descriptor())) {
      return false;
    }
    owned = std::unique_ptr<ProtoType>(new ProtoType());
    value = owned.get();
    return pybind11_protobuf::PyProtoCopyToCProto(src, owned.get());
  }

  // ensure_owned ensures that the owned member contains a copy of the
  // ::google::protobuf::Message.
  void ensure_owned() {
    if (value && !owned) {
      owned = std::unique_ptr<ProtoType>(value->New());
      *owned = *value;
      value = owned.get();
    }
  }

  const ProtoType *value;
  std::unique_ptr<ProtoType> owned;
};

template <>
struct proto_caster_load_impl<::google::protobuf::Message> {
  using ProtoType = ::google::protobuf::Message;

  bool load(pybind11::handle src, bool convert) {
    if (src.is_none()) {
      value = nullptr;
      return true;
    }

    // Attempt to use the PyProto_API to get an underlying C++ message pointer
    // from the object.
    value = pybind11_protobuf::PyProtoGetCppMessagePointer(src);
    if (value && value->GetDescriptor() && value->GetDescriptor()->file() &&
        value->GetDescriptor()->file()->pool() ==
            ::google::protobuf::DescriptorPool::generated_pool()) {
      // Only messages in the same generated_pool() can be referenced directly.
      return true;
    }

    // `src` is not a C++ proto instance from the generated_pool,
    // so create a compatible native C++ proto.
    auto descriptor_name = pybind11_protobuf::PyProtoDescriptorName(src);
    if (!descriptor_name) {
      return false;
    }
    owned.reset(static_cast<ProtoType *>(
        pybind11_protobuf::AllocateCProtoFromPythonSymbolDatabase(
            src, *descriptor_name)
            .release()));
    value = owned.get();
    return pybind11_protobuf::PyProtoCopyToCProto(src, owned.get());
  }

  // ensure_owned ensures that the owned member contains a copy of the
  // ::google::protobuf::Message.
  void ensure_owned() {
    if (value && !owned) {
      owned = std::unique_ptr<ProtoType>(value->New());
      owned->CopyFrom(*value);
      value = owned.get();
    }
  }

  const ::google::protobuf::Message *value;
  std::unique_ptr<::google::protobuf::Message> owned;
};

struct fast_cpp_cast_impl {
  inline static pybind11::handle cast_impl(::google::protobuf::Message *src,
                                           pybind11::return_value_policy policy,
                                           pybind11::handle parent,
                                           bool is_const) {
    if (src == nullptr) return pybind11::none().release();

#if PYBIND11_PROTOBUF_UNSAFE
    if (is_const &&
        (policy == pybind11::return_value_policy::reference ||
         policy == pybind11::return_value_policy::reference_internal)) {
      throw pybind11::type_error(
          "Cannot return a const reference to a ::google::protobuf::Message derived "
          "type.  Consider setting return_value_policy::copy in the "
          "pybind11 def().");
    }
#else
    // references are inherently unsafe, so convert them to copies.
    if (policy == pybind11::return_value_policy::reference ||
        policy == pybind11::return_value_policy::reference_internal) {
      policy = pybind11::return_value_policy::copy;
    }
#endif

    return pybind11_protobuf::GenericFastCppProtoCast(src, policy, parent,
                                                      is_const);
  }
};

struct native_cast_impl {
  inline static pybind11::handle cast_impl(::google::protobuf::Message *src,
                                           pybind11::return_value_policy policy,
                                           pybind11::handle parent,
                                           bool is_const) {
    if (src == nullptr) return pybind11::none().release();

    // references are inherently unsafe, so convert them to copies.
    if (policy == pybind11::return_value_policy::reference ||
        policy == pybind11::return_value_policy::reference_internal) {
      policy = pybind11::return_value_policy::copy;
    }

    return pybind11_protobuf::GenericProtoCast(src, policy, parent, false);
  }
};

// pybind11 type_caster specialization for c++ protocol buffer types.
template <typename ProtoType, typename CastBase>
struct proto_caster : public proto_caster_load_impl<ProtoType>,
                      protected CastBase {
 private:
  using Loader = proto_caster_load_impl<ProtoType>;
  using CastBase::cast_impl;
  using Loader::ensure_owned;
  using Loader::owned;
  using Loader::value;

 public:
  static constexpr auto name = pybind11::detail::_<ProtoType>();

  // cast converts from C++ -> Python
  //
  // return_value_policy handling differs from the behavior for
  // py::class_-wrapped objects because because protocol buffer objects usually
  // need to be copied across the C++/python boundary as they contain internal
  // pointers which are unsafe to modify. See:
  // https://pybind11.readthedocs.io/en/stable/advanced/functions.html#return-value-policies
  static pybind11::handle cast(ProtoType &&src,
                               pybind11::return_value_policy policy,
                               pybind11::handle parent) {
    return cast_impl(&src, pybind11::return_value_policy::move, parent, false);
  }

  static pybind11::handle cast(const ProtoType *src,
                               pybind11::return_value_policy policy,
                               pybind11::handle parent) {
    std::unique_ptr<const ProtoType> wrapper;
    if (policy == pybind11::return_value_policy::automatic ||
        policy == pybind11::return_value_policy::automatic_reference) {
      policy = pybind11::return_value_policy::copy;
    } else if (policy == pybind11::return_value_policy::take_ownership) {
      wrapper.reset(src);
    }
    return cast_impl(const_cast<ProtoType *>(src), policy, parent, true);
  }

  static pybind11::handle cast(ProtoType *src,
                               pybind11::return_value_policy policy,
                               pybind11::handle parent) {
    std::unique_ptr<ProtoType> wrapper;
    if (policy == pybind11::return_value_policy::automatic_reference) {
      policy = pybind11::return_value_policy::copy;
    } else if (policy == pybind11::return_value_policy::automatic ||
               policy == pybind11::return_value_policy::take_ownership) {
      policy = pybind11::return_value_policy::take_ownership;
      wrapper.reset(src);
    }
    return cast_impl(src, policy, parent, false);
  }

  static pybind11::handle cast(ProtoType const &src,
                               pybind11::return_value_policy policy,
                               pybind11::handle parent) {
    if (policy == pybind11::return_value_policy::automatic ||
        policy == pybind11::return_value_policy::automatic_reference) {
      policy = pybind11::return_value_policy::copy;
    }
    return cast_impl(const_cast<ProtoType *>(&src), policy, parent, true);
  }

  static pybind11::handle cast(ProtoType &src,
                               pybind11::return_value_policy policy,
                               pybind11::handle parent) {
    if (policy == pybind11::return_value_policy::automatic ||
        policy == pybind11::return_value_policy::automatic_reference) {
      policy = pybind11::return_value_policy::copy;
    }
    return cast_impl(&src, policy, parent, false);
  }

  std::unique_ptr<ProtoType> as_unique_ptr() {
    ensure_owned();
    return std::move(owned);
  }

  // PYBIND11_TYPE_CASTER
  explicit operator const ProtoType *() { return value; }
  explicit operator const ProtoType &() {
    if (!value) throw pybind11::reference_cast_error();
    return *value;
  }
  explicit operator ProtoType &&() && {
    if (!value) throw pybind11::reference_cast_error();
    ensure_owned();
    return std::move(*owned);
  }

#if PYBIND11_PROTOBUF_UNSAFE
  // The following unsafe conversions are not enabled:
  explicit operator ProtoType *() { return const_cast<ProtoType *>(value); }
  explicit operator ProtoType &() {
    if (!value) throw pybind11::reference_cast_error();
    return *const_cast<ProtoType *>(value);
  }
#endif

  // cast_op_type determines which operator overload to call for a given c++
  // input parameter type.
  // clang-format off
  template <typename T_>
  using cast_op_type =
      std::conditional_t<
          std::is_same<std::remove_reference_t<T_>, const ProtoType *>::value,
              const ProtoType *,
      std::conditional_t<
          std::is_same<
              std::remove_reference_t<T_>, ProtoType *>::value, ProtoType *,
      std::conditional_t<
          std::is_same<T_, const ProtoType &>::value, const ProtoType &,
      std::conditional_t<std::is_same<T_, ProtoType &>::value, ProtoType &,
      /*default is T&&*/ T_>>>>;
  // clang-format on
};

}  // namespace pybind11_protobuf

#endif  // PYBIND11_PROTOBUF_PROTO_CASTER_IMPL_H_
