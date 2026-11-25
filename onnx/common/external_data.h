// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE::checker {

// These lexical checks perform no filesystem access.
ONNX_API std::filesystem::path check_external_data_location_syntax(
    const std::string& base_dir,
    const std::string& location,
    const std::string& tensor_name);
ONNX_API void check_external_data_location_in_model_dir(
    const std::string& base_dir,
    const std::filesystem::path& location,
    const std::string& tensor_name);

// Pins one directory identity. Sidecar components are opened relative to that
// handle without following links. The caller owns returned CRT descriptors.
class ONNX_API ExternalDataContext final {
 public:
  explicit ExternalDataContext(const std::string& base_dir);
  ~ExternalDataContext();
  ExternalDataContext(const ExternalDataContext&) = delete;
  ExternalDataContext& operator=(const ExternalDataContext&) = delete;
  int64_t open(const std::string& location, const std::string& tensor_name, bool read_only) const;
  int64_t open_model_file(const std::string& location) const;
  int64_t open_existing_model_file_for_write(const std::string& location) const;
  int64_t create_model_file(const std::string& location) const;
  std::uintmax_t file_size(const std::string& location, const std::string& tensor_name) const;

 private:
  enum class ModelFileOpenMode { kRead, kWriteExisting, kCreateNew };
  int64_t open_model_file_impl(const std::string& location, ModelFileOpenMode mode) const;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace ONNX_NAMESPACE::checker
