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
std::string fix_sep(std::string path) {
  std::string out = path;
  normalize_separator(out);
  return out;
}
} // namespace

TEST(PathTest, CleanRelativePathTest) {
  // Already normal.
  EXPECT_EQ(clean_relative_path("abc"), fix_sep("abc"));
  EXPECT_EQ(clean_relative_path("abc/def"), fix_sep("abc/def"));
  EXPECT_EQ(clean_relative_path("a/b/c"), fix_sep("a/b/c"));
  EXPECT_EQ(clean_relative_path("."), fix_sep("."));
  EXPECT_EQ(clean_relative_path(".."), fix_sep(".."));
  EXPECT_EQ(clean_relative_path("../.."), fix_sep("../.."));
  EXPECT_EQ(clean_relative_path("../../abc"), fix_sep("../../abc"));
  // Remove leading slash
  EXPECT_EQ(clean_relative_path("/abc"), fix_sep("abc"));
  EXPECT_EQ(clean_relative_path("/"), fix_sep("."));
  // Remove trailing slash
  EXPECT_EQ(clean_relative_path("abc/"), fix_sep("abc"));
  EXPECT_EQ(clean_relative_path("abc/def/"), fix_sep("abc/def"));
  EXPECT_EQ(clean_relative_path("a/b/c/"), fix_sep("a/b/c"));
  EXPECT_EQ(clean_relative_path("./"), fix_sep("."));
  EXPECT_EQ(clean_relative_path("../"), fix_sep(".."));
  EXPECT_EQ(clean_relative_path("../../"), fix_sep("../.."));
  EXPECT_EQ(clean_relative_path("/abc/"), fix_sep("abc"));
  // Remove doubled slash
  EXPECT_EQ(clean_relative_path("abc//def//ghi"), fix_sep("abc/def/ghi"));
  EXPECT_EQ(clean_relative_path("//abc"), fix_sep("abc"));
  EXPECT_EQ(clean_relative_path("///abc"), fix_sep("abc"));
  EXPECT_EQ(clean_relative_path("//abc//"), fix_sep("abc"));
  EXPECT_EQ(clean_relative_path("abc//"), fix_sep("abc"));
  // Remove . elements
  EXPECT_EQ(clean_relative_path("abc/./def"), fix_sep("abc/def"));
  EXPECT_EQ(clean_relative_path("/./abc/def"), fix_sep("abc/def"));
  EXPECT_EQ(clean_relative_path("abc/."), fix_sep("abc"));
  // Remove .. elements
  EXPECT_EQ(clean_relative_path("abc/def/ghi/../jkl"), fix_sep("abc/def/jkl"));
  EXPECT_EQ(clean_relative_path("abc/def/../ghi/../jkl"), fix_sep("abc/jkl"));
  EXPECT_EQ(clean_relative_path("abc/def/.."), fix_sep("abc"));
  EXPECT_EQ(clean_relative_path("abc/def/../.."), fix_sep("."));
  EXPECT_EQ(clean_relative_path("/abc/def/../.."), fix_sep("."));
  EXPECT_EQ(clean_relative_path("abc/def/../../.."), fix_sep(".."));
  EXPECT_EQ(clean_relative_path("/abc/def/../../.."), fix_sep(".."));
  EXPECT_EQ(clean_relative_path("abc/def/../../../ghi/jkl/../../../mno"), fix_sep("../../mno"));
  EXPECT_EQ(clean_relative_path("/../abc"), fix_sep("../abc"));
  // Combinations
  EXPECT_EQ(clean_relative_path("abc/./../def"), fix_sep("def"));
  EXPECT_EQ(clean_relative_path("abc//./../def"), fix_sep("def"));
  EXPECT_EQ(clean_relative_path("abc/../../././../def"), fix_sep("../../def"));
}

} // namespace Test
} // namespace ONNX_NAMESPACE
