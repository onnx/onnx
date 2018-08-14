// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include <cstdarg>
#include <cstdio>

#include "onnx/common/assertions.h"

namespace ONNX_NAMESPACE {

std::string barf(const char* fmt, ...) {
  char msg[2048];
  va_list args;

  va_start(args, fmt);
  vsnprintf(msg, 2048, fmt, args);
  va_end(args);

  return std::string(msg);
}

void throw_assert_error(std::string& msg) {
  throw assert_error(msg);
}

void throw_tensor_error(std::string& msg) {
  throw tensor_error(msg);
}

} // namespace ONNX_NAMESPACE
