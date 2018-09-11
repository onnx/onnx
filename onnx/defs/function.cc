// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/function.h"
#include "onnx/checker.h"
#include "onnx/defs/schema.h"
#include "onnx/string_utils.h"

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

Status FunctionBuilderRegistry::Register(
    const FunctionBuilder& function_builder) {
  std::lock_guard<std::mutex> lock(mutex_);
  function_builders.push_back(function_builder);
  return Status::OK();
}

// Get functions for specific domain.
Status FunctionBuilderRegistry::GetFunctions(
    const std::string& domain,
    /*out*/
    std::multimap<std::string, std::unique_ptr<FunctionProto>>* function_set)
    const {
  if (nullptr == function_set) {
    return Common::Status(
        Common::CHECKER,
        Common::INVALID_ARGUMENT,
        "function_set should not be nullptr.");
  }

  for (auto func_builder : function_builders) {
    if (func_builder.GetDomain() != domain) {
      continue;
    }
    std::unique_ptr<FunctionProto> function_proto;
    auto status = func_builder.GetBuildFunction()(&function_proto);
    if (!status.IsOK()) {
      return status;
    }

    CheckerContext ctx;
    std::unordered_map<std::string, int> op_set;
    auto version_range =
        OpSchemaRegistry::DomainToVersionRange::Instance().Map().at(
            func_builder.GetDomain());
    if (function_proto->since_version() > version_range.second ||
        function_proto->since_version() < version_range.first) {
      fail_check("Invalid function version in '", function_proto->name(), "'");
    }
    op_set.insert({func_builder.GetDomain(), (int)function_proto->since_version()});
    ctx.set_opset_imports(op_set);
    ctx.set_is_main_graph(false);
    LexicalScopeContext lex_ctx;
    try {
      check_function(*function_proto, ctx, lex_ctx);
    } catch (ValidationError& ex) {
      return Common::Status(
          Common::CHECKER, Common::INVALID_PROTOBUF, ex.what());
    }

    auto& func_name = function_proto->name();
    // Check no op version conflicts.
    auto range = function_set->equal_range(func_name);
    for (auto i = range.first; i != range.second; ++i) {
      auto version = i->second->since_version();
      if (function_proto->since_version() == version) {
        // There's already a function with same name/since_version registered.
        return Common::Status(
            Common::CHECKER,
            Common::FAIL,
            ONNX_NAMESPACE::MakeString(
                "A function (",
                func_name,
                ") with version (",
                version,
                ") has already been registered."));
      }
    }
    function_set->emplace(func_name, std::move(function_proto));
  }

  return Common::Status::OK();
}

FunctionBuilderRegistry& FunctionBuilderRegistry::OnnxInstance() {
  static FunctionBuilderRegistry func_builder_registry;
  return func_builder_registry;
}
} // namespace ONNX_NAMESPACE
