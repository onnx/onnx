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

TEST(PathTest, CleanRelativePathTest) {
  std::list<std::pair<std::string, std::string>> test_cases = {
      // Already normal.
      {"abc", "abc"},
      {"abc/def", "abc/def"},
      {"a/b/c", "a/b/c"},
      {".", "."},
      {"..", ".."},
      {"../..", "../.."},
      {"../../abc", "../../abc"},

      // Remove leading slash
      {"/abc", "abc"},
      {"/", "."},

      // Remove trailing slash
      {"abc/", "abc"},
      {"abc/def/", "abc/def"},
      {"a/b/c/", "a/b/c"},
      {"./", "."},
      {"../", ".."},
      {"../../", "../.."},
      {"/abc/", "abc"},

      // Remove doubled slash
      {"abc//def//ghi", "abc/def/ghi"},
      {"//abc", "abc"},
      {"///abc", "abc"},
      {"//abc//", "abc"},
      {"abc//", "abc"},

      // Remove . elements
      {"abc/./def", "abc/def"},
      {"/./abc/def", "abc/def"},
      {"abc/.", "abc"},

      // Remove .. elements
      {"abc/def/ghi/../jkl", "abc/def/jkl"},
      {"abc/def/../ghi/../jkl", "abc/jkl"},
      {"abc/def/..", "abc"},
      {"abc/def/../..", "."},
      {"/abc/def/../..", "."},
      {"abc/def/../../..", ".."},
      {"/abc/def/../../..", ".."},
      {"abc/def/../../../ghi/jkl/../../../mno", "../../mno"},
      {"/../abc", "../abc"},

      // Combinations
      {"abc/./../def", "def"},
      {"abc//./../def", "def"},
      {"abc/../../././../def", "../../def"},
  };

  for (std::pair<std::string, std::string>& test_case : test_cases) {
    std::string path = test_case.first;
    std::string expected = test_case.second;
    // Normalize separator in case of windows tests.
    normalize_separator(expected);

    EXPECT_EQ(clean_relative_path(path), expected) << "Invalid relative path returned for " << path;
  }
}

} // namespace Test
} // namespace ONNX_NAMESPACE
