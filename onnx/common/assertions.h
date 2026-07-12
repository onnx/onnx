// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <stdexcept>
#include <string>

#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {

struct assert_error : public std::runtime_error {
 public:
  explicit assert_error(const std::string& msg) : runtime_error(msg) {}
};

struct tensor_error : public assert_error {
 public:
  explicit tensor_error(const std::string& msg) : assert_error(msg) {}
};

[[noreturn]] void throw_assert_error(const std::string& /*msg*/);

[[noreturn]] void throw_tensor_error(const std::string& /*msg*/);

} // namespace ONNX_NAMESPACE

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define _ONNX_EXPECT(x, y) (__builtin_expect((x), (y)))
#else
#define _ONNX_EXPECT(x, y) (x)
#endif

// The message arguments are concatenated with std::stringstream (via MakeString),
// so std::string, std::string_view, and numbers are all supported directly without
// format specifiers or .c_str().
#define ONNX_ASSERT(cond)                                                                                           \
  if (_ONNX_EXPECT(!(cond), 0)) { /* NOLINT(readability-simplify-boolean-expr) */                                   \
    std::string error_msg =                                                                                         \
        ::ONNX_NAMESPACE::MakeString(__FILE__, ":", __LINE__, ": ", __func__, ": Assertion `", #cond, "` failed."); \
    throw_assert_error(error_msg);                                                                                  \
  }

#define ONNX_ASSERTM(cond, ...)                                                                      \
  /* NOLINTNEXTLINE */                                                                               \
  if (_ONNX_EXPECT(!(cond), 0)) {                                                                    \
    std::string error_msg = ::ONNX_NAMESPACE::MakeString(                                            \
        __FILE__, ":", __LINE__, ": ", __func__, ": Assertion `", #cond, "` failed: ", __VA_ARGS__); \
    throw_assert_error(error_msg);                                                                   \
  }

#define TENSOR_ASSERTM(cond, ...)                                                                    \
  if (_ONNX_EXPECT(!(cond), 0)) {                                                                    \
    std::string error_msg = ::ONNX_NAMESPACE::MakeString(                                            \
        __FILE__, ":", __LINE__, ": ", __func__, ": Assertion `", #cond, "` failed: ", __VA_ARGS__); \
    throw_tensor_error(error_msg);                                                                   \
  }
