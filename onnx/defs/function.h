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

typedef Common::Status (*BuildFunction)(FunctionProto*);

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

  // Register function proto builder.
  // This is not thread-safe.
  Status Register(const FunctionBuilder& function_builder);

  // Initialize function protos hold by this registry.
  // This is not thread-safe.
  Status Init();

  // Get all function protos.
  // This should be called after Init() function call.
  const std::multimap<std::string, FunctionProto>& GetFunctions() const;

  static FunctionBuilderRegistry& OnnxInstance();

 private:
  std::vector<FunctionBuilder> function_builders;
  std::multimap<std::string, FunctionProto> function_set_;
  std::unique_ptr<Status> init_status_;
};

#define ONNX_FUNCTION(function_builder) \
  ONNX_FUNCTION_UNIQ_HELPER(__COUNTER__, function_builder)

#define ONNX_FUNCTION_UNIQ_HELPER(counter, function_builder) \
  ONNX_FUNCTION_UNIQ(counter, function_builder)

#define ONNX_FUNCTION_UNIQ(counter, function_builder)         \
  static Common::Status function_builder_##counter##_status = \
      FunctionBuilderRegistry::OnnxInstance().Register(function_builder);

// Example to register a function.
//Common::Status BuildFc(FunctionProto* func_proto) {
//  if (nullptr == func_proto) {
//    return Status(
//        Common::CHECKER,
//        Common::INVALID_ARGUMENT,
//        "func_proto should not be nullptr.");
//  }
//
//  func_proto->set_name("FC");
//  // set function inputs.
//  // set function outputs.
//  // set function attributes.
//  // set function description.
//  // set function body (nodes).
//
//  return Status::OK();
//}
//
//ONNX_FUNCTION(FunctionBuilder().SetDomain("").SetBuildFunction(BuildFc));

} // namespace ONNX_NAMESPACE
