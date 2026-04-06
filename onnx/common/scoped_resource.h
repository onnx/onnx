// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
// Usage:
//   using ScopedFd = ScopedResource<-1, close_fd>;
//   ScopedFd guard(fd);          // destructor calls close_fd(fd)
//   int raw = guard.release();   // relinquish ownership
template <auto Invalid, void (*Close)(decltype(Invalid))>
class ScopedResource {
  using T = decltype(Invalid);
  T val_;

 public:
  explicit ScopedResource(T v) : val_(v) {}
  ~ScopedResource() {
    if (val_ != Invalid) {
      Close(val_);
    }
  }
  T get() const {
    return val_;
  }
  T release() {
    T tmp = val_;
    val_ = Invalid;
    return tmp;
  }
  ScopedResource(const ScopedResource&) = delete;
  ScopedResource& operator=(const ScopedResource&) = delete;
};

// Platform-specific type aliases.

#ifdef _WIN32
inline void close_handle(HANDLE h) {
  CloseHandle(h);
}
using ScopedHandle = ScopedResource<INVALID_HANDLE_VALUE, close_handle>;
#endif

inline void close_fd(int fd) {
#ifdef _WIN32
  _close(fd);
#else
  close(fd);
#endif
}
using ScopedFd = ScopedResource<-1, close_fd>;

} // namespace ONNX_NAMESPACE
