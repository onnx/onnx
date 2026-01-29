// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "onnx/checker.h"

namespace fs = std::filesystem;

namespace ONNX_NAMESPACE {
namespace Test {

TEST(CHECKER, ValidDataLocationTest) {
  EXPECT_THROW(
      ONNX_NAMESPACE::checker::resolve_external_data_location("localfolder", "..", "tensor_name"),
      ONNX_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(
      ONNX_NAMESPACE::checker::resolve_external_data_location("localfolder", "/usr/any", "tensor_name"),
      ONNX_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(
      ONNX_NAMESPACE::checker::resolve_external_data_location("localfolder", "./sub/example", "tensor_name"),
      ONNX_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(
      ONNX_NAMESPACE::checker::resolve_external_data_location("localfolder", "sub/example", "tensor_name"),
      ONNX_NAMESPACE::checker::ValidationError);
}

TEST(CHECKER, ValidDataLocationSymLinkTest) {
  fs::path tempDir = fs::temp_directory_path() / "symlink_test-%%%%%%";
  fs::create_directories(tempDir);
  fs::path target = tempDir / "model.data";
  fs::path link = tempDir / "link.data";
  fs::create_symlink(target, link);
#ifdef WIN32
  std::string location = link.u8string();
#else
  std::string location = link.c_str();
#endif
  EXPECT_THROW(
      ONNX_NAMESPACE::checker::resolve_external_data_location("localfolder", location, "tensor_name"),
      ONNX_NAMESPACE::checker::ValidationError);
  fs::remove(link);
  fs::remove(target);
  fs::remove(tempDir);
}

} // namespace Test
} // namespace ONNX_NAMESPACE
