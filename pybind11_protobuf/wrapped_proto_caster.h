#ifndef PYBIND11_PROTOBUF_WRAPPED_PROTO_CASTER_H_
#define PYBIND11_PROTOBUF_WRAPPED_PROTO_CASTER_H_

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include <google/protobuf/message.h>
//#include "absl/status/statusor.h"
//#include "absl/types/optional.h"
#include "proto_cast_util.h"
#include "proto_caster_impl.h"

// pybind11::type_caster<> specialization for ::google::protobuf::Message types using
// pybind11_protobuf::WrappedProto<> wrappers. When passing from C++ to
// python a copy is always made. When passing from python to C++ a copy is
// made for mutable and by-value types, but not for const reference types.
//
// This is intended as a migration aid until such a time as proto_casters.h
// can be replaced with native_proto_caster.h.
//
// Example:
//
// #include <pybind11/pybind11.h>
// #include "pybind11_protobuf/wrapped_proto_casters.h"
//
// MyMessage GetMessage() { ... }
//
// PYBIND11_MODULE(my_module, m) {
//   pybind11_protobuf::ImportWrappedProtoCasters();
//
//   using pybind11_protobuf::WithWrappedProtos;
//
//   m.def("get_message", WithWrappedProtos(&GetMessage));
// }
//
// Under the covers, WithWrappedProtos(...) maps by-const or by-value
// proto::Message subtype method argument and return types to a transparent
// wrapper, WrappedProto<T, Kind> types which are then handled by a specialized
// pybind11::type_caster.
//
// Mutable types are not automatically converted, as the always-copy
// semantics may be misleading for this conversion, and callers likely need to
// make adjustments to the caller or return types of such code. Of particluar
// note, builder-style classes may be better implemented independently in
// python rather than attempting to wrap C++ counterparts.
//
//
// Users of wrapped_proto_caster based extensions need dependencies on:
// deps = [ "@com_google_protobuf//:protobuf_python" ]
//

namespace pybind11_protobuf {

// Imports modules for protobuf conversion. This not thread safe and
// is required to be called from a PYBIND11_MODULE definition before use.
inline void ImportWrappedProtoCasters() { InitializePybindProtoCastUtil(); }

/// Tag types for WrappedProto specialization.
enum WrappedProtoKind : int { kConst, kValue, kMutable };

/// WrappedProto<T, kind> wraps a ::google::protobuf::Message subtype, exposing implicit
/// constructors and implicit cast operators. The ConstTag variant allows
/// conversion to const; the MutableTag allows mutable conversion and enables
/// the type-caster specialization to enforce that a copy is made.
///
/// WrappedProto<T, kMutable> ALWAYS creates a copy.
/// WrappedProto<T, kValue> ALWAYS creates a copy.
/// WrappedProto<T, kConst> may either copy or share the underlying class.
template <typename ProtoType, WrappedProtoKind Kind>
struct WrappedProto;

template <typename ProtoType>
struct WrappedProto<ProtoType, WrappedProtoKind::kMutable> {
  using type = ProtoType;
  static constexpr WrappedProtoKind kind = WrappedProtoKind::kMutable;
  ProtoType* proto;

  WrappedProto(ProtoType* p) : proto(p) {}
  WrappedProto(ProtoType& p) : proto(&p) {}
  ProtoType* get() noexcept { return proto; }

  operator ProtoType*() noexcept { return proto; }
  operator ProtoType&() {
    if (!proto) throw pybind11::reference_cast_error();
    return *proto;
  }
};

template <typename ProtoType>
struct WrappedProto<ProtoType, WrappedProtoKind::kConst> {
  using type = ProtoType;
  static constexpr WrappedProtoKind kind = WrappedProtoKind::kConst;
  const ProtoType* proto;

  WrappedProto(const ProtoType* p) : proto(p) {}
  WrappedProto(const ProtoType& p) : proto(&p) {}
  ProtoType* get() noexcept { return const_cast<ProtoType*>(proto); }

  operator const ProtoType*() const noexcept { return proto; }
  operator const ProtoType&() const {
    if (!proto) throw pybind11::reference_cast_error();
    return *proto;
  }
};

template <typename ProtoType>
struct WrappedProto<ProtoType, WrappedProtoKind::kValue> {
  using type = ProtoType;
  static constexpr WrappedProtoKind kind = WrappedProtoKind::kValue;
  ProtoType proto;

  WrappedProto(ProtoType&& p) : proto(std::move(p)) {}
  ProtoType* get() noexcept { return &proto; }

  operator ProtoType&&() && noexcept { return std::move(proto); }
};

// pybind11 type_caster specialization for WrappedProto<T>
// c++ protocol buffer types.
template <typename WrappedProtoType>
struct wrapped_proto_caster : public pybind11_protobuf::proto_caster_load_impl<
                                  typename WrappedProtoType::type>,
                              protected pybind11_protobuf::native_cast_impl {
  using Base = pybind11_protobuf::proto_caster_load_impl<
      typename WrappedProtoType::type>;
  using ProtoType = typename WrappedProtoType::type;
  using Base::ensure_owned;
  using Base::owned;
  using Base::value;
  using pybind11_protobuf::native_cast_impl::cast_impl;

 public:
  static constexpr auto name = pybind11::detail::_<WrappedProtoType>();

  // cast converts from C++ -> Python
  static pybind11::handle cast(WrappedProtoType src,
                               pybind11::return_value_policy policy,
                               pybind11::handle parent) {
    if (src.get() == nullptr) return pybind11::none().release();
    std::unique_ptr<ProtoType> owned;

    if (WrappedProtoType::kind == WrappedProtoKind::kValue) {
      // When the proto is by-value, it's always possible to move the contents
      policy = pybind11::return_value_policy::move;
    } else {
      // Otherwise take a copy by default.
      if (policy == pybind11::return_value_policy::automatic ||
          policy == pybind11::return_value_policy::automatic_reference) {
        policy = pybind11::return_value_policy::copy;
      } else if (policy == pybind11::return_value_policy::take_ownership) {
        owned.reset(const_cast<ProtoType*>(src.get()));
      }
    }
    return cast_impl(
        src.get(), policy, parent,
        /*is_const*/ WrappedProtoKind::kConst == WrappedProtoType::kind);
  }

  // PYBIND11_TYPE_CASTER
  template <WrappedProtoKind K>
  typename std::enable_if<(K == WrappedProtoKind::kValue),
                          WrappedProtoType>::type
  convert() {
    // WrappedProto<T, kValue> ALWAYS creates a copy.
    if (owned) {
      return std::move(*owned);
    } else {
      return ProtoType(*value);
    }
  }

  template <WrappedProtoKind K>
  typename std::enable_if<(K == WrappedProtoKind::kMutable),
                          WrappedProtoType>::type
  convert() {
    // WrappedProto<T, kMutable> ALWAYS creates a copy.
    ensure_owned();
    return owned.get();
  }

  template <WrappedProtoKind K>
  typename std::enable_if<(K == WrappedProtoKind::kConst),
                          WrappedProtoType>::type
  convert() {
    return value;
  }

  explicit operator WrappedProtoType() {
    return this->template convert<WrappedProtoType::kind>();
  }

  template <typename T_>
  using cast_op_type = WrappedProtoType;
};

/// Support for wrapping methods using std::vector<ProtoType> with
/// WithWrappedProto.
///
/// In the single-proto cases, a transparent wrapper, WrappedProto<Proto, Kind>,
/// is used to select the native proto pybind11 type_caster<> and avoid ODR
/// issues. That mechanism doesn't work very well for containers, since to
/// convert from a std::vector<Wrapper<X>> to std::vector<X> involves a
/// copy/move, even if the wrapper is trivial.
///
/// This workaround takes a similar tack to the WrappedProto<> transparent
/// wrapper: Create a transparent wrapper of std::vector<T>, and implement a
/// custom type_caster to delegate the conversion of the inner Proto type to
/// native_proto_caster, avoiding the default pybind11 list_caster conversion
/// and make_type_caster<> selection.  WrapHelper applies the rewrite to the
/// functions.
///
/// NOTE: This always copies values, similar to WrappedProto<Kind, kValue>.

/// WrappedProtoVector<T> wraps a std::vector<T>. It is used to add std::vector
/// support for WithWrappedProtos by forwarding the internal T to the native
/// proto casters.
template <typename ProtoType>
struct WrappedProtoVector {
  std::vector<ProtoType> protos;

  WrappedProtoVector() = default;

  WrappedProtoVector(WrappedProtoVector&&) = default;
  WrappedProtoVector(const WrappedProtoVector&) = default;
  WrappedProtoVector& operator=(WrappedProtoVector&&) = default;
  WrappedProtoVector& operator=(const WrappedProtoVector&) = default;

  WrappedProtoVector(std::vector<ProtoType>&& p) : protos(std::move(p)) {}

  operator std::vector<ProtoType>&&() && noexcept { return std::move(protos); }
};

// type_caster<> implementation for WrappedProtoVector. See
// pybind11::list_caster.
template <typename ProtoType>
struct wrapped_proto_vector_caster {
  static constexpr auto name =
      (pybind11::detail::const_name("List[") +
       pybind11::detail::_<ProtoType>() + pybind11::detail::const_name("]"));

  // cast converts from Python -> C++
  bool load(pybind11::handle src, bool convert) {
    using load_impl = pybind11_protobuf::proto_caster_load_impl<ProtoType>;

    if (!pybind11::isinstance<pybind11::sequence>(src) ||
        pybind11::isinstance<pybind11::bytes>(src) ||
        pybind11::isinstance<pybind11::str>(src)) {
      return false;
    }
    auto s = pybind11::reinterpret_borrow<pybind11::sequence>(src);
    value.protos.clear();
    value.protos.reserve(s.size());
    for (auto it : s) {
      load_impl conv;
      if (!conv.load(it, convert)) {
        return false;
      }
      if (conv.owned) {
        value.protos.emplace_back(std::move(*conv.owned));
      } else {
        value.protos.emplace_back(*conv.value);
      }
    }
    return true;
  }

  // cast converts from C++ -> Python
  static pybind11::handle cast(WrappedProtoVector<ProtoType> src,
                               pybind11::return_value_policy policy,
                               pybind11::handle parent) {
    pybind11::list l(src.protos.size());
    ssize_t index = 0;
    for (auto&& value : src.protos) {
      auto value_ = pybind11::reinterpret_steal<pybind11::object>(
          pybind11_protobuf::native_cast_impl::cast_impl(
              &value, pybind11::return_value_policy::move, parent,
              /*is_const*/ false));
      if (!value_) {
        return pybind11::handle();
      }
      PyList_SET_ITEM(l.ptr(), index++,
                      value_.release().ptr());  // steals a reference
    }
    return l.release();
  }

  explicit operator WrappedProtoVector<ProtoType>&&() && {
    return std::move(value);
  }

  template <typename T_>
  using cast_op_type = WrappedProtoVector<ProtoType>&&;

  WrappedProtoVector<ProtoType> value;
};

// Implementation details for WithWrappedProtos
namespace impl {

template <typename T>
using intrinsic_t =
    typename pybind11::detail::intrinsic_t<std::remove_reference_t<T>>;

template <typename T>
constexpr WrappedProtoKind DetectKind() {
  using no_ref_t = std::remove_reference_t<T>;
  using base_t = intrinsic_t<no_ref_t>;

  if (std::is_same<no_ref_t, const base_t*>::value) {
    return WrappedProtoKind::kConst;
  } else if (std::is_same<no_ref_t, base_t*>::value) {
    return WrappedProtoKind::kMutable;
  } else if (std::is_same<T, const base_t&>::value) {
    return WrappedProtoKind::kConst;
  } else if (std::is_same<T, base_t&>::value) {
    return WrappedProtoKind::kMutable;
  } else {
    return WrappedProtoKind::kValue;
  }
}

template <typename T, typename SFINAE = void>
struct WrapHelper {
  using type = T;
};

// Rewrite Proto to WrappedProto<Proto, Kind>
template <typename ProtoType>
struct WrapHelper<ProtoType,  //
                  std::enable_if_t<std::is_base_of<
                      ::google::protobuf::Message, intrinsic_t<ProtoType>>::value>> {
  static constexpr auto kKind = DetectKind<ProtoType>();
  static_assert(kKind != WrappedProtoKind::kMutable,
                "WithWrappedProtos() does not support mutable ::google::protobuf::Message "
                "parameters.");

  using type = WrappedProto<intrinsic_t<ProtoType>, kKind>;
};

// absl::optional is also sometimes used as a wrapper for optional protos.
template <typename ProtoType>
struct WrapHelper<absl::optional<ProtoType>,  //
                  std::enable_if_t<std::is_base_of<
                      ::google::protobuf::Message, intrinsic_t<ProtoType>>::value>> {
  static constexpr auto kKind = DetectKind<ProtoType>();
  static_assert(kKind != WrappedProtoKind::kMutable,
                "WithWrappedProtos() does not support mutable ::google::protobuf::Message "
                "parameters.");

  using type = absl::optional<WrappedProto<intrinsic_t<ProtoType>, kKind>>;
};

// absl::StatusOr is a common wrapper so also propagate WrappedProto into
// StatusOr types.  When wrapped by WithWrappedProtos, ensure that
// pybind11_abseil/status_casters.h is included and follow the instructions
// there.
template <typename ProtoType>
struct WrapHelper<absl::StatusOr<ProtoType>,  //
                  std::enable_if_t<std::is_base_of<
                      ::google::protobuf::Message, intrinsic_t<ProtoType>>::value>> {
  static constexpr auto kKind = DetectKind<ProtoType>();
  static_assert(kKind != WrappedProtoKind::kMutable,
                "WithWrappedProtos() does not support mutable ::google::protobuf::Message "
                "parameters.");

  using type = absl::StatusOr<WrappedProto<intrinsic_t<ProtoType>, kKind>>;
};

// Also rewrite std::vector<Proto> to WrappedProtoVector<Proto> when passed by
// const reference or by value.
template <typename ProtoType>
struct WrapHelper<std::vector<ProtoType>,  //
                  std::enable_if_t<std::is_base_of<
                      ::google::protobuf::Message, intrinsic_t<ProtoType>>::value>> {
  using type = WrappedProtoVector<ProtoType>;
};
template <typename ProtoType>
struct WrapHelper<const std::vector<ProtoType>&,  //
                  std::enable_if_t<std::is_base_of<
                      ::google::protobuf::Message, intrinsic_t<ProtoType>>::value>> {
  using type = WrappedProtoVector<ProtoType>;
};

// And absl::StatusOr<std::vector<Proto>>
template <typename ProtoType>
struct WrapHelper<absl::StatusOr<std::vector<ProtoType>>,  //
                  std::enable_if_t<std::is_base_of<
                      ::google::protobuf::Message, intrinsic_t<ProtoType>>::value>> {
  using type = absl::StatusOr<WrappedProtoVector<ProtoType>>;
};

/// FunctionInvoker/ConstMemberInvoker/MemberInvoker are internal functions
/// that expose calls with equivalent signatures, replacing ::google::protobuf::Message
/// derived types with WrappedProto<T, ...> types.
template <typename F, typename>
struct FunctionInvoker;

template <typename F, typename R, typename... Args>
struct FunctionInvoker<F, R(Args...)> {
  F f;
  typename WrapHelper<R>::type operator()(
      typename WrapHelper<Args>::type... args) const {
    return f(std::forward<decltype(args)>(args)...);
  }
};

template <typename F, typename R, typename... Args>
struct FunctionInvoker<F, R(Args...) const> {
  F f;
  typename WrapHelper<R>::type operator()(
      typename WrapHelper<Args>::type... args) const {
    return f(std::forward<decltype(args)>(args)...);
  }
};

template <typename C, typename R, typename... Args>
struct ConstMemberInvoker {
  static_assert(!std::is_base_of<::google::protobuf::Message, C>::value,
                "WithWrappedProtos should not be used on methods generated by "
                "the protocol buffer compiler");

  R (C::*f)(Args...) const;
  typename WrapHelper<R>::type operator()(
      const C& c, typename WrapHelper<Args>::type... args) const {
    return (c.*f)(std::forward<decltype(args)>(args)...);
  }
};

template <typename C, typename R, typename... Args>
struct MemberInvoker {
  static_assert(!std::is_base_of<::google::protobuf::Message, C>::value,
                "WithWrappedProtos should not be used on methods generated by "
                "the protocol buffer compiler");

  R (C::*f)(Args...);
  typename WrapHelper<R>::type operator()(
      C& c, typename WrapHelper<Args>::type... args) const {
    return (c.*f)(std::forward<decltype(args)>(args)...);
  }
};

/// Helper to extract a member function signature. A lambda or other functor's
/// operator() signature can be extracted, when present, via:
///   LambdaSignature<decltype(&T::operator())>::type
template <typename>
struct LambdaSignature;

template <typename U, typename T>
struct LambdaSignature<U T::*> {
  using type = U;
};

}  // namespace impl

/// WithWrappedProtos(...) wraps a function type and converts any
/// ::google::protobuf::Message derived types into a WrappedProto<...> of the same
/// underlying proto2 type.
template <typename F, int&... ExplicitArgumentBarrier,
          typename S = impl::LambdaSignature<decltype(&F::operator())>>
auto WithWrappedProtos(F f) ->
    typename impl::FunctionInvoker<F, typename S::type> {
  return {std::move(f)};
}

template <typename R, typename... Args>
impl::FunctionInvoker<R (*)(Args...), R(Args...)>  //
WithWrappedProtos(R (*f)(Args...)) {
  return {f};
}

template <typename R, typename C, typename... Args>
impl::MemberInvoker<C, R, Args...>  //
WithWrappedProtos(R (C::*f)(Args...)) {
  return {f};
}

template <typename R, typename C, typename... Args>
impl::ConstMemberInvoker<C, R, Args...>  //
WithWrappedProtos(R (C::*f)(Args...) const) {
  return {f};
}

}  // namespace pybind11_protobuf
namespace pybind11 {
namespace detail {

// pybind11 type_caster<> specialization for c++ protocol buffer types using
// inheritance from pybind11_protobuf::wrapped_proto_caster<>.
template <typename ProtoType, pybind11_protobuf::WrappedProtoKind Kind>
struct type_caster<
    pybind11_protobuf::WrappedProto<ProtoType, Kind>,
    std::enable_if_t<std::is_base_of<::google::protobuf::Message, ProtoType>::value>>
    : public pybind11_protobuf::wrapped_proto_caster<
          pybind11_protobuf::WrappedProto<ProtoType, Kind>> {};

// pybind11 type_caster<> specialization for WrappedProtoVector<proto>.
template <typename ProtoType>
struct type_caster<
    pybind11_protobuf::WrappedProtoVector<ProtoType>,
    std::enable_if_t<std::is_base_of<::google::protobuf::Message, ProtoType>::value>>
    : public pybind11_protobuf::wrapped_proto_vector_caster<ProtoType> {};

}  // namespace detail
}  // namespace pybind11

#endif  // PYBIND11_PROTOBUF_WRAPPED_PROTO_CASTER_H_
