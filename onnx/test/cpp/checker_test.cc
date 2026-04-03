// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#ifdef _WIN32
#include <io.h>
#define CLOSE_FD _close
#else
#include <unistd.h>
#define CLOSE_FD close
#endif

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

TEST(CHECKER, OpenExternalDataTest) {
#ifndef ONNX_NO_EXCEPTIONS
  // Use a UTF-8 directory name (raw bytes for C++20 char8_t compat).
  fs::path dir = fs::temp_directory_path() / "\xe6\xa8\xa1\xe5\x9e\x8b_onnx_open_test";
  fs::remove_all(dir);
  fs::create_directories(dir);

  // Read existing file (UTF-8 filename)
  std::string utf8_file = "\xe3\x83\x86\xe3\x83\xb3\xe3\x82\xbd\xe3\x83\xab.bin";
  {
    std::ofstream ofs(dir / utf8_file, std::ios::binary);
    ofs << "data";
  }
  auto raw = ONNX_NAMESPACE::checker::open_external_data(dir.string(), utf8_file, "t", true);
  EXPECT_GE(raw, 0);
  int fd = static_cast<int>(raw);
  char buf[4];
  EXPECT_EQ(read(fd, buf, 4), 4);
  EXPECT_EQ(std::string(buf, 4), "data");
  CLOSE_FD(fd);

  // Write creates new file
  raw = ONNX_NAMESPACE::checker::open_external_data(dir.string(), "new.bin", "t", false);
  EXPECT_GE(raw, 0);
  CLOSE_FD(static_cast<int>(raw));
  EXPECT_TRUE(fs::exists(dir / "new.bin"));

  // Reject invalid locations
  EXPECT_THROW(
      ONNX_NAMESPACE::checker::open_external_data(dir.string(), "missing.bin", "t", true),
      ONNX_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(
      ONNX_NAMESPACE::checker::open_external_data(dir.string(), "/etc/passwd", "t", true),
      ONNX_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(
      ONNX_NAMESPACE::checker::open_external_data(dir.string(), "../escape", "t", true),
      ONNX_NAMESPACE::checker::ValidationError);
  EXPECT_THROW(
      ONNX_NAMESPACE::checker::open_external_data(dir.string(), ".", "t", false),
      ONNX_NAMESPACE::checker::ValidationError);
  // UTF-8 traversal
  EXPECT_THROW(
      ONNX_NAMESPACE::checker::open_external_data(
          dir.string(), "../\xe3\x83\x86\xe3\x83\xb3\xe3\x82\xbd\xe3\x83\xab.bin", "t", true),
      ONNX_NAMESPACE::checker::ValidationError);

#ifndef _WIN32
  // Symlink to outside directory
  fs::path outside = fs::temp_directory_path() / "onnx_open_ext_outside";
  fs::remove_all(outside);
  fs::create_directories(outside);
  {
    std::ofstream ofs(outside / "secret.data", std::ios::binary);
    ofs << "secret";
  }
  fs::create_symlink(outside / "secret.data", dir / "link.data");
  EXPECT_THROW(
      ONNX_NAMESPACE::checker::open_external_data(dir.string(), "link.data", "t", true),
      ONNX_NAMESPACE::checker::ValidationError);
  fs::remove_all(outside);
#endif

  fs::remove_all(dir);
#endif
}

} // namespace Test
} // namespace ONNX_NAMESPACE
