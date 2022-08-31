/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <list>
#include <utility>
#include "gtest/gtest.h"

#include "onnx/common/path.h"

using namespace ONNX_NAMESPACE;

namespace ONNX_NAMESPACE {
namespace Test {
namespace {
template <typename STRING>
STRING fix_sep(const STRING& path) {
  STRING out = path;
  normalize_separator(out);
  return out;
}

template <typename STRING>
void expect(const STRING& actual, const STRING& expected, const std::string& error_message) {
  // Test clean_relative_path and normalize_separator
  EXPECT_EQ(clean_relative_path(actual), fix_sep(expected)) << error_message;
}
void expect_string_and_wstring(const std::string& actual, const std::string& expected) {
  // Test string
  expect(actual, expected, "string path mismatch.");
#ifdef _WIN32
  // Test wstring with utf8str_to_wstring
  expect(utf8str_to_wstring(actual), utf8str_to_wstring(expected), "wstring path mismatch.");
#endif _WIN32
}
} // namespace

TEST(PathTest, CleanRelativePathTest) {
  // Already normal.
  expect_string_and_wstring("abc", "abc");
  expect_string_and_wstring("abc/def", "abc/def");
  expect_string_and_wstring("a/b/c", "a/b/c");
  expect_string_and_wstring(".", ".");
  expect_string_and_wstring("..", "..");
  expect_string_and_wstring("../..", "../..");
  expect_string_and_wstring("../../abc", "../../abc");
  // Remove leading slash
  expect_string_and_wstring("/abc", "abc");
  expect_string_and_wstring("/", ".");
  // Remove trailing slash
  expect_string_and_wstring("abc/", "abc");
  expect_string_and_wstring("abc/def/", "abc/def");
  expect_string_and_wstring("a/b/c/", "a/b/c");
  expect_string_and_wstring("./", ".");
  expect_string_and_wstring("../", "..");
  expect_string_and_wstring("../../", "../..");
  expect_string_and_wstring("/abc/", "abc");
  // Remove doubled slash
  expect_string_and_wstring("abc//def//ghi", "abc/def/ghi");
  expect_string_and_wstring("//abc", "abc");
  expect_string_and_wstring("///abc", "abc");
  expect_string_and_wstring("//abc//", "abc");
  expect_string_and_wstring("abc//", "abc");
  // Remove . elements
  expect_string_and_wstring("abc/./def", "abc/def");
  expect_string_and_wstring("/./abc/def", "abc/def");
  expect_string_and_wstring("abc/.", "abc");
  // Remove .. elements
  expect_string_and_wstring("abc/def/ghi/../jkl", "abc/def/jkl");
  expect_string_and_wstring("abc/def/../ghi/../jkl", "abc/jkl");
  expect_string_and_wstring("abc/def/..", "abc");
  expect_string_and_wstring("abc/def/../..", ".");
  expect_string_and_wstring("/abc/def/../..", ".");
  expect_string_and_wstring("abc/def/../../..", "..");
  expect_string_and_wstring("/abc/def/../../..", "..");
  expect_string_and_wstring("abc/def/../../../ghi/jkl/../../../mno", "../../mno");
  expect_string_and_wstring("/../abc", "../abc");
  // Combinations
  expect_string_and_wstring("abc/./../def", "def");
  expect_string_and_wstring("abc//./../def", "def");
  expect_string_and_wstring("abc/../../././../def", "../../def");
}

} // namespace Test
} // namespace ONNX_NAMESPACE
