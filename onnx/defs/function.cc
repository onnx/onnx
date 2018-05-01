// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/checker.h"
#include "onnx/defs/function.h"

namespace ONNX_NAMESPACE {
using namespace checker;
FunctionBuilder& FunctionBuilder::SetDomain(const std::string& domain) {
  domain_ = domain;
  return *this;
}

const std::string& FunctionBuilder::GetDomain() const {
  return domain_;
}

FunctionBuilder& FunctionBuilder::SetBuildFunction(BuildFunction build_func) {
  build_func_ = build_func;
  return *this;
}

BuildFunction FunctionBuilder::GetBuildFunction() const {
  return build_func_;
}

void FunctionBuilderRegistry::Register(
    const FunctionBuilder& function_builder) {
  std::lock_guard<std::mutex> lock(mutex_);
  function_builders.push_back(function_builder);
}

// Get functions for specific domain.
Status FunctionBuilderRegistry::GetFunctions(
    const std::string& domain,
    /*out*/
    std::multimap<std::string, std::shared_ptr<FunctionProto>>* function_set)
    const {
  if (nullptr == function_set) {
    return Common::Status(
        Common::OPSCHEMA,
        Common::INVALID_ARGUMENT,
        "function_set should not be nullptr.");
  }

  for (auto func_builder : function_builders) {
    if (func_builder.GetDomain() != domain) {
      continue;
    }
    std::shared_ptr<FunctionProto> function_proto;
    auto status = func_builder.GetBuildFunction()(&function_proto);
    if (!status.IsOK()) {
      return status;
    }

    CheckerContext ctx;
    LexicalScopeContext lex_ctx;
    status = check_function(*function_proto, ctx, lex_ctx);
    if (!status.IsOK()) {
      return status;
    }

    auto& func_name = function_proto->name();
    // Check no op version conflicts.
    auto range = function_set->equal_range(func_name);
    for (auto i = range.first; i != range.second; ++i) {
      auto version = i->second->since_version();
      if (function_proto->since_version() == version) {
        // There's already a function with same name/since_version registered.
        return Common::Status(
            Common::OPSCHEMA,
            Common::FAIL,
            "A function (" + func_name + ") with version (" +
                std::to_string(version) + ") has already been registered.");
      }
    }
    function_set->emplace(func_name, function_proto);
  }

  return Common::Status::OK();
}

FunctionBuilderRegistry& FunctionBuilderRegistry::OnnxInstance() {
  static FunctionBuilderRegistry func_builder_registry;

  //func_builder_registry.Register(FunctionBuilder().SetDomain("").SetBuildFunction(BuildFc));

  return func_builder_registry;
}

//Common::Status BuildFc(std::shared_ptr<FunctionProto>* func_proto) {
//	if (nullptr == func_proto) {
//		return Status(Common::OPSCHEMA, Common::INVALID_ARGUMENT, "func_proto should not be nullptr.");
//	}
//
//	func_proto->reset(new FunctionProto);
//	auto& func = **func_proto;
//	func.set_name("FC");
//	// set function inputs.
//	// set function outputs.
//	// set function attributes.
//	// set function description.
//	// set function body (nodes).
//
//	return Status::OK();
//}

} // namespace ONNX_NAMESPACE
