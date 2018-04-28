// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <xstring>
#include <mutex>
#include <vector>

#include "onnx/common/status.h"
#include "onnx/onnx-operators_pb.h"

namespace ONNX_NAMESPACE {
using namespace Common;

typedef Common::Status (*BuildFunction)(std::shared_ptr<FunctionProto>*);

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

  void Register(const FunctionBuilder& function_builder);

  // Get functions for specific domain.
  Status GetFunctions(
      const std::string& domain,
      /*out*/
      std::multimap<std::string, std::shared_ptr<FunctionProto>>* function_set)
      const;

  static FunctionBuilderRegistry& OnnxInstance();

 private:
  std::vector<FunctionBuilder> function_builders;
  std::mutex mutex_;
}; // namespace ONNX_NAMESPACE

} // namespace ONNX_NAMESPACE
