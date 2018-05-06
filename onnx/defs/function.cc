// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/function.h"
#include "onnx/checker.h"
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
  function_builders.push_back(function_builder);
  return Status::OK();
}

const std::multimap<std::string, FunctionProto>&
FunctionBuilderRegistry::GetFunctions() const {
  return function_set_;
}

// Get functions for specific domain.
Status FunctionBuilderRegistry::Init() {
  if (nullptr != init_status_) {
    return *init_status_;
  }

  init_status_.reset(new Status());
  for (auto func_builder : function_builders) {
    FunctionProto function_proto;
	*init_status_ = func_builder.GetBuildFunction()(&function_proto);
    if (!init_status_->IsOK()) {
      return *init_status_;
    }

    CheckerContext ctx;
    LexicalScopeContext lex_ctx;
    try {
      check_function(function_proto, ctx, lex_ctx);
    } catch (ValidationError& ex) {
      init_status_.reset(
          new Status(Common::CHECKER, Common::INVALID_PROTOBUF, ex.what()));
      return *init_status_;
    }

    auto& func_name = function_proto.name();
    // Check no op version conflicts.
    auto range = function_set_.equal_range(func_name);
    for (auto i = range.first; i != range.second; ++i) {
      auto version = i->second.since_version();
      if (function_proto.since_version() == version) {
        // There's already a function with same name/since_version registered.
        init_status_.reset(new Status(
            Common::CHECKER,
            Common::FAIL,
            ONNX_NAMESPACE::MakeString(
                "A function (",
                func_name,
                ") with version (",
                version,
                ") has already been registered.")));
        return *init_status_;
      }
    }
    function_set_.emplace(func_name, std::move(function_proto));
  }

  return *init_status_;
}

FunctionBuilderRegistry& FunctionBuilderRegistry::OnnxInstance() {
  static FunctionBuilderRegistry func_builder_registry;
  return func_builder_registry;
}

} // namespace ONNX_NAMESPACE
