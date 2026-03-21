// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "onnx/checker.h"

namespace fs = std::filesystem;

namespace ONNX_NAMESPACE {
namespace Test {

TEST(CHECKER, ValidDataLocationTest) {
#ifndef ONNX_NO_EXCEPTIONS
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
#endif
}

TEST(CHECKER, ValidDataLocationSymLinkTest) {
#if !defined(ONNX_NO_EXCEPTIONS) && !defined(_WIN32)
  // Use a temp directory as the base_dir (simulating the model directory).
  // We pass a relative filename as the location so that the absolute-path
  // rejection (checker.cc line 986) is NOT triggered, and the is_symlink()
  // check (checker.cc line 1016) is actually exercised.
  fs::path modelDir = fs::temp_directory_path() / "onnx_symlink_checker_test";
  fs::remove_all(modelDir);
  fs::create_directories(modelDir);

  // Create a regular target file so the symlink has a valid target.
  fs::path target = modelDir / "target.data";
  {
    std::ofstream ofs(target);
    ofs << "test data";
  }

  // Create a symlink pointing to the target file.
  fs::path link = modelDir / "link.data";
  fs::create_symlink(target, link);

  // Pass relative filename "link.data" — the checker resolves it to
  // modelDir/link.data and should reject it because it is a symlink.
  EXPECT_THROW(
      ONNX_NAMESPACE::checker::resolve_external_data_location(modelDir.string(), "link.data", "tensor_name"),
      ONNX_NAMESPACE::checker::ValidationError);

  fs::remove_all(modelDir);
#endif
}

TEST(CHECKER, ValidDataLocationParentDirSymLinkTest) {
#if !defined(ONNX_NO_EXCEPTIONS) && !defined(_WIN32)
  // Test that symlinks in parent directory components are detected.
  // A location like "symlink_subdir/real_file.data" where symlink_subdir
  // is a symlink to an outside directory should be rejected by the
  // canonical path containment check in checker.cc.
  fs::path modelDir = fs::temp_directory_path() / "onnx_parent_symlink_test";
  fs::remove_all(modelDir);
  fs::create_directories(modelDir);

  // Create a target directory outside the model directory.
  fs::path outsideDir = fs::temp_directory_path() / "onnx_outside_target";
  fs::remove_all(outsideDir);
  fs::create_directories(outsideDir);

  // Create a real file in the outside directory.
  fs::path targetFile = outsideDir / "secret.data";
  {
    std::ofstream ofs(targetFile);
    ofs << "sensitive data";
  }

  // Create a directory symlink inside modelDir pointing outside.
  fs::path symlinkSubdir = modelDir / "subdir";
  fs::create_directory_symlink(outsideDir, symlinkSubdir);

  // "subdir/secret.data" is a relative path where "subdir" is a symlink.
  // The canonical path resolves outside modelDir, so this should be rejected.
  EXPECT_THROW(
      ONNX_NAMESPACE::checker::resolve_external_data_location(modelDir.string(), "subdir/secret.data", "tensor_name"),
      ONNX_NAMESPACE::checker::ValidationError);

  fs::remove_all(modelDir);
  fs::remove_all(outsideDir);
#endif
}

} // namespace Test
} // namespace ONNX_NAMESPACE
