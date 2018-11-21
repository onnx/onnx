// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "onnx/common/constants.h"
#include "onnx/common/status.h"
#include "onnx/onnx-operators_pb.h"

namespace ONNX_NAMESPACE {

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

class IFunctionBuilderRegistry {
 public:
  virtual ~IFunctionBuilderRegistry() = default;

  virtual const FunctionProto* GetFunction(
      const std::string& func_name,
      const int maxInclusiveVersion,
      const std::string& domain = ONNX_DOMAIN) const = 0;
};

class FunctionBuilderRegistry : public IFunctionBuilderRegistry {
 public:
  FunctionBuilderRegistry() = default;

  Common::Status Register(const FunctionBuilder& function_builder);

  // Get functions for specific domain.
  Common::Status GetFunctions(
      const std::string& domain,
      /*out*/
      std::multimap<std::string, const FunctionProto*>* function_set) const;

  const FunctionProto* GetFunction(
      const std::string& func_name,
      const int maxInclusiveVersion,
      const std::string& domain = ONNX_DOMAIN) const override;

  static FunctionBuilderRegistry& OnnxInstance();

 private:
  std::vector<FunctionBuilder> function_builders;
  std::unordered_map<
      std::string,
      std::multimap<std::string, std::unique_ptr<FunctionProto>>>
      domain_functions_map;
  std::mutex mutex_;
};

template <typename T>
FunctionBuilder GetFunctionBuilder();

#define ONNX_FUNCTION_BUILDER_CLASS_NAME(domain, ver, name) \
  name##_##domain##_ver##ver

#define ONNX_FUNCTION_BUILD(name, ver, build_func) \
  ONNX_FUNCTION_BUILD_HELPER(name, Onnx, ONNX_DOMAIN, ver, build_func)

#define ONNX_FUNCTION_BUILD_HELPER(name, domain, domain_str, ver, build_func) \
  class ONNX_FUNCTION_BUILDER_CLASS_NAME(domain, ver, name);                  \
  template <>                                                                 \
  FunctionBuilder                                                             \
  GetFunctionBuilder<ONNX_FUNCTION_BUILDER_CLASS_NAME(domain, ver, name)>() { \
    return build_func;                                                        \
  }

#define ONNX_FUNCTION(function_builder) \
  ONNX_FUNCTION_UNIQ_HELPER(__COUNTER__, function_builder)

#define ONNX_FUNCTION_UNIQ_HELPER(counter, function_builder) \
  ONNX_FUNCTION_UNIQ(counter, function_builder)

#define ONNX_FUNCTION_UNIQ(counter, function_builder)         \
  static Common::Status function_builder_##counter##_status = \
      FunctionBuilderRegistry::OnnxInstance().Register(function_builder);

inline void RegisterOneFunctionBuilder(FunctionBuilder&& func_builder) {
  ONNX_FUNCTION(func_builder);
}

// Registers all function builder of a given operator set
template <class T>
void RegisterFunctionBuilder() {
  T::ForEachFunctionBuilder(RegisterOneFunctionBuilder);
};

// Helper function to expand a function node given the function proto
void FunctionExpandHelper(
    const NodeProto& node,
    const FunctionProto& func,
    GraphProto& g,
    const std::string& node_prefix = "");

// Example to register a function.
// Common::Status BuildFc(std::unique_ptr<FunctionProto>* func_proto) {
//  if (nullptr == func_proto) {
//    return Status(
//        Common::CHECKER,
//        Common::INVALID_ARGUMENT,
//        "func_proto should not be nullptr.");
//  }
//
//  func_proto->reset(new FunctionProto);
//  auto& func = **func_proto;
//  func.set_name("FC");
//   set function inputs.
//   set function outputs.
//   set function attributes.
//   set function description.
//   set function body (nodes).
//
//  return Status::OK();
//}
//
// ONNX_FUNCTION_BUILD(Name, Ver,
// FunctionBuilder().SetDomain("").SetBuildFunction(BuildFc));

} // namespace ONNX_NAMESPACE
