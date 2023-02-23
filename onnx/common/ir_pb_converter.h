/*
 * SPDX-License-Identifier: Apache-2.0
 */

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once
#include "onnx/common/common.h"
#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {

class ConvertError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;

  explicit ConvertError(const std::string& message) : std::runtime_error(message) {}

  const char* what() const noexcept override {
    if (!expanded_message_.empty()) {
      return expanded_message_.c_str();
    }
    return std::runtime_error::what();
  }

  void AppendContext(const std::string& context) {
    expanded_message_ = MakeString(std::runtime_error::what(), "\n\n==> Context: ", context);
  }

 private:
  std::string expanded_message_;
};

#define fail_convert(...) ONNX_THROW_EX(ConvertError(MakeString(__VA_ARGS__)));

void ExportModelProto(
    ModelProto* p_m,
    const std::shared_ptr<GraphBase>& g,
    std::vector<std::shared_ptr<FunctionIR>>& local_functions);

std::unique_ptr<GraphBase> ImportModelProto(
    const ModelProto& mp,
    std::vector<std::shared_ptr<FunctionIR>>& local_functions);

ModelProto PrepareOutput(const ModelProto& mp_in);

void assertNonNull(const std::shared_ptr<GraphBase>& g);
} // namespace ONNX_NAMESPACE
