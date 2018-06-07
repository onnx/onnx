// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "onnx/common/status.h"
#include "onnx/onnx-operators_pb.h"

namespace ONNX_NAMESPACE {
using namespace Common;

typedef Common::Status (*BuildFunction)(std::unique_ptr<FunctionProto>*);

class FunctionBuilder {
 public:
  FunctionBuilder& SetDomain(const std::string& domain);
  const std::string& GetDomain() const;
  FunctionBuilder& SetBuildFunction(BuildFunction build_func);
  BuildFunction GetBuildFunction() const;

 private:
  std::string domain_;
  BuildFunction build_func_;
};

class FunctionBuilderRegistry {
 public:
  FunctionBuilderRegistry() = default;

  Status Register(const FunctionBuilder& function_builder);

  // Get functions for specific domain.
  Status GetFunctions(
      const std::string& domain,
      /*out*/
      std::multimap<std::string, std::unique_ptr<FunctionProto>>* function_set)
      const;

  static FunctionBuilderRegistry& OnnxInstance();

 private:
  std::vector<FunctionBuilder> function_builders;
  std::mutex mutex_;
};

#define ONNX_FUNCTION(function_builder) \
  ONNX_FUNCTION_UNIQ_HELPER(__COUNTER__, function_builder)

#define ONNX_FUNCTION_UNIQ_HELPER(counter, function_builder) \
  ONNX_FUNCTION_UNIQ(counter, function_builder)

#define ONNX_FUNCTION_UNIQ(counter, function_builder)         \
  static Common::Status function_builder_##counter##_status = \
      FunctionBuilderRegistry::OnnxInstance().Register(function_builder);

} // namespace ONNX_NAMESPACE
