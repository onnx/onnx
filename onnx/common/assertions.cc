/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/common/assertions.h"
#include <assert.h>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include "onnx/common/common.h"
#define BUFFER_MAX_SIZE 2048

namespace ONNX_NAMESPACE {

std::string barf(const char* fmt, ...) {
  char msg[BUFFER_MAX_SIZE];
  va_list args;
  // To resolve potential vulnerability issue for vsnprintf, add an assert to make sure the quantity is reasonable.
  assert(strlen(fmt) <= BUFFER_MAX_SIZE && "The string length for vsnprintf is larger than the buffer size 2048.");
  va_start(args, fmt);
  vsnprintf(msg, BUFFER_MAX_SIZE, fmt, args);
  va_end(args);
  return std::string(msg);
}

void throw_assert_error(std::string& msg) {
  ONNX_THROW_EX(assert_error(msg));
}

void throw_tensor_error(std::string& msg) {
  ONNX_THROW_EX(tensor_error(msg));
}

} // namespace ONNX_NAMESPACE
