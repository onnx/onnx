// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace ONNX_NAMESPACE {

// Returns true on overflow. Uses __builtin on GCC/Clang, manual check on MSVC.
// Both a and b must be non-negative; safe_dim_product enforces this by checking
// each dim before calling, and the accumulated result stays non-negative because
// we abort on overflow.
inline bool checked_mul_overflow(int64_t a, int64_t b, int64_t* result) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_mul_overflow(a, b, result);
#else
  if (a > 0 && b > std::numeric_limits<int64_t>::max() / a) {
    return true;
  }
  *result = a * b;
  return false;
#endif
}

// Safe product of dims. Calls on_error(const char*) on negative dim or overflow.
// on_error must not return (i.e. must throw or abort).
template <typename DimsContainer, typename ErrorHandler>
[[nodiscard]] inline int64_t safe_dim_product(const DimsContainer& dims, ErrorHandler on_error) {
  int64_t result = 1;
  for (auto d : dims) {
    int64_t dim = static_cast<int64_t>(d);
    if (dim < 0) {
      on_error("Negative dimension value");
      return result; // unreachable if on_error throws; guards against misuse
    }
    if (checked_mul_overflow(result, dim, &result)) {
      on_error("Tensor dimension product overflow");
      return result;
    }
  }
  return result;
}

// Iterator-pair overload for subranges (avoids container copy).
template <typename Iter, typename ErrorHandler>
[[nodiscard]] inline int64_t safe_dim_product(Iter begin, Iter end, ErrorHandler on_error) {
  int64_t result = 1;
  for (auto it = begin; it != end; ++it) {
    int64_t dim = static_cast<int64_t>(*it);
    if (dim < 0) {
      on_error("Negative dimension value");
      return result;
    }
    if (checked_mul_overflow(result, dim, &result)) {
      on_error("Tensor dimension product overflow");
      return result;
    }
  }
  return result;
}

// Safe cast from int64_t to size_t. Calls on_error if the value exceeds
// size_t range (relevant for 32-bit platforms where size_t is 32 bits).
// value must be non-negative (callers ensure this via prior overflow checks).
template <typename ErrorHandler>
[[nodiscard]] inline size_t safe_cast_to_size(int64_t value, ErrorHandler on_error) {
  if (static_cast<uint64_t>(value) > std::numeric_limits<size_t>::max()) {
    on_error("Value too large for this platform");
  }
  return static_cast<size_t>(value);
}

} // namespace ONNX_NAMESPACE
