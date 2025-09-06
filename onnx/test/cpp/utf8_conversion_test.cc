// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef _WIN32
#include <string>

#include "catch2/catch_test_macros.hpp"
#include "onnx/common/path.h"
namespace ONNX_NAMESPACE::Test {

TEST_CASE("UTF8Test", "[WideStringConvertion]") {
  std::string utf8_str(u8"世界，你好！");
  REQUIRE(ONNX_NAMESPACE::wstring_to_utf8str(ONNX_NAMESPACE::utf8str_to_wstring(utf8_str)) == utf8_str);
}

TEST_CASE("UTF8Test", "[TryConvertUTF8]") {
  std::string utf8_str(u8"世界，你好！");
  auto wstr = ONNX_NAMESPACE::utf8str_to_wstring(utf8_str);
  auto wstr2 = ONNX_NAMESPACE::utf8str_to_wstring(
      std::string(reinterpret_cast<const char*>(wstr.c_str()), sizeof(std::wstring::value_type) * wstr.size()), true);
  REQUIRE(wstr == wstr2);
}
} // namespace ONNX_NAMESPACE::Test
#endif
