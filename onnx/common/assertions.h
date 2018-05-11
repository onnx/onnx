// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <exception>
#include <string>

namespace ONNX_NAMESPACE {

struct assert_error final : public std::exception {
private:
  const std::string msg_;
public:
  explicit assert_error(std::string msg) : msg_(std::move(msg)) {}
  const char* what() const noexcept override { return msg_.c_str(); }
};

[[noreturn]]
void barf(const char *fmt, ...);

} // namespace ONNX_NAMESPACE

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
#define ONNX_EXPECT(x, y) (__builtin_expect((x), (y)))
#else
#define ONNX_EXPECT(x, y) (x)
#endif

#define ONNX_ASSERT(cond) \
  if (ONNX_EXPECT(!(cond), 0)) { \
    ::ONNX_NAMESPACE::barf("%s:%u: %s: Assertion `%s` failed.", __FILE__, __LINE__, __func__, #cond); \
  }

// The trailing ' ' argument is a hack to deal with the extra comma when ... is empty.
// Another way to solve this is ##__VA_ARGS__ in _ONNX_ASSERTM, but this is a non-portable
// extension we shouldn't use.
#define ONNX_ASSERTM(...) _ONNX_ASSERTM(__VA_ARGS__, " ")

// Note: msg must be a string literal
#define _ONNX_ASSERTM(cond, msg, ...) \
  if (ONNX_EXPECT(!(cond), 0)) { \
    ::ONNX_NAMESPACE::barf("%s:%u: %s: Assertion `%s` failed: " msg, __FILE__, __LINE__, __func__, #cond, __VA_ARGS__); \
  }

#define ONNX_EXPECTM(...) _ONNX_EXPECTM(__VA_ARGS__, " ")

// Note: msg must be a string literal
#define _ONNX_EXPECTM(cond, msg, ...) \
  if (ONNX_EXPECT(!(cond), 0)) { \
    ::ONNX_NAMESPACE::barf("%s:%u: %s: " msg, __FILE__, __LINE__, __func__, __VA_ARGS__); \
  }
