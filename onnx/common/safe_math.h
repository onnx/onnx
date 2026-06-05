// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>

namespace ONNX_NAMESPACE {

// Returns true on overflow for all signed int64 values.
// GCC/Clang use the compiler builtin. The MSVC fallback handles INT64_MIN
// explicitly (it cannot be negated) and uses abs-based division for the rest.
inline bool checked_mul_overflow(int64_t a, int64_t b, int64_t* result) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_mul_overflow(a, b, result);
#else
  if (a == 0 || b == 0) {
    *result = 0;
    return false;
  }
  // INT64_MIN cannot be negated safely; its only non-overflowing multiplier is 1.
  if (a == std::numeric_limits<int64_t>::min()) {
    if (b == 1) {
      *result = a;
      return false;
    }
    return true;
  }
  if (b == std::numeric_limits<int64_t>::min()) {
    if (a == 1) {
      *result = b;
      return false;
    }
    return true;
  }
  const int64_t abs_a = a < 0 ? -a : a;
  const int64_t abs_b = b < 0 ? -b : b;
  if (abs_a > std::numeric_limits<int64_t>::max() / abs_b) {
    return true;
  }
  *result = a * b;
  return false;
#endif
}

// Returns true on overflow for all signed int64 values.
// Uses unsigned addition (no UB) and detects overflow via sign-bit XOR.
// The cast back to int64_t is implementation-defined in C++17 but gives
// two's-complement on every MSVC target; mandated by the standard from C++20.
// Overflow iff a and b have the same sign but the result has the opposite sign.
inline bool checked_add_overflow(int64_t a, int64_t b, int64_t* result) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_add_overflow(a, b, result);
#else
  const auto ur = static_cast<uint64_t>(a) + static_cast<uint64_t>(b);
  *result = static_cast<int64_t>(ur);
  return ((a ^ *result) & (b ^ *result)) < 0;
#endif
}

// Returns true on overflow for all signed int64 values.
// Uses unsigned subtraction (no UB) and detects overflow via sign-bit XOR.
// Overflow iff a and b have different signs and the result has a different sign from a.
inline bool checked_sub_overflow(int64_t a, int64_t b, int64_t* result) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_sub_overflow(a, b, result);
#else
  const auto ur = static_cast<uint64_t>(a) - static_cast<uint64_t>(b);
  *result = static_cast<int64_t>(ur);
  return ((a ^ b) & (a ^ *result)) < 0;
#endif
}

// Safe product of dims over an iterator range. Calls on_error(const char*) on
// negative dim or overflow. on_error must not return (i.e. must throw or abort).
template <typename Iter, typename ErrorHandler>
[[nodiscard]] inline int64_t safe_dim_product(Iter begin, Iter end, ErrorHandler on_error) {
  int64_t result = 1;
  for (auto it = begin; it != end; ++it) {
    auto dim = static_cast<int64_t>(*it);
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

// Container overload — delegates to the iterator-pair version.
template <typename DimsContainer, typename ErrorHandler>
[[nodiscard]] inline int64_t safe_dim_product(const DimsContainer& dims, ErrorHandler on_error) {
  return safe_dim_product(std::begin(dims), std::end(dims), on_error);
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
