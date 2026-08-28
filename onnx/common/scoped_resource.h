// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ONNX_NAMESPACE {

// Generic RAII guard for resources identified by a sentinel "invalid" value and
// a free-function closer.  Non-copyable, non-movable.
//
// Traits must provide:
//   using type = ...;             // the resource's value type
//   static type invalid();        // the sentinel "invalid" value
//   static void close(type);      // the closer
//
// The sentinel is obtained through Traits::invalid() rather than a non-type
// template argument because some platform sentinels (e.g. Windows'
// INVALID_HANDLE_VALUE, which reinterpret-casts -1 to a pointer) are not
// valid converted constant expressions for a pointer-typed non-type template
// parameter, even though permissive compilers accept them.
//
// Usage:
//   using ScopedFd = ScopedResource<FdTraits>;
//   ScopedFd guard(fd);          // destructor calls FdTraits::close(fd)
//   int raw = guard.release();   // relinquish ownership
template <typename Traits>
class ScopedResource {
  using T = typename Traits::type;
  T val_;

 public:
  explicit ScopedResource(T v) : val_(v) {}
  ~ScopedResource() {
    if (val_ != Traits::invalid()) {
      Traits::close(val_);
    }
  }
  T get() const {
    return val_;
  }
  T release() {
    T tmp = val_;
    val_ = Traits::invalid();
    return tmp;
  }
  ScopedResource(const ScopedResource&) = delete;
  ScopedResource& operator=(const ScopedResource&) = delete;
};

// Platform-specific type aliases.

#ifdef _WIN32
struct HandleTraits {
  using type = HANDLE;
  static HANDLE invalid() {
    return INVALID_HANDLE_VALUE;
  }
  static void close(HANDLE h) {
    CloseHandle(h);
  }
};
using ScopedHandle = ScopedResource<HandleTraits>;
#endif

struct FdTraits {
  using type = int;
  static constexpr int invalid() {
    return -1;
  }
  static void close(int fd) {
#ifdef _WIN32
    _close(fd);
#else
    ::close(fd);
#endif
  }
};
using ScopedFd = ScopedResource<FdTraits>;

// RAII guard that invokes a callable on destruction (scope exit).
template <typename F>
class ScopeExit {
  F fn_;

 public:
  explicit ScopeExit(F fn) : fn_(std::move(fn)) {}
  ~ScopeExit() noexcept {
    static_assert(std::is_nothrow_invocable_v<F&>, "ScopeExit callable must be noexcept");
    fn_();
  }
  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;
};

} // namespace ONNX_NAMESPACE
