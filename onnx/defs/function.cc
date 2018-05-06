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
  function_builders_.push_back(function_builder);
  return Status::OK();
}

const FuncName_Domain_Version_Proto_Map& FunctionBuilderRegistry::GetFunctions()
    const {
  return function_set_;
}

// Get functions for specific domain.
Status FunctionBuilderRegistry::Init() {
  if (nullptr != init_status_) {
    return *init_status_;
  }

  init_status_.reset(new Status());
  for (auto func_builder : function_builders_) {
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
    auto& func_domain = func_builder.GetDomain();
    int func_version = static_cast<int>(function_proto.since_version());

    auto ver_range_it = domain_version_range_.find(func_domain);
    if (ver_range_it == domain_version_range_.end()) {
      init_status_.reset(new Status(
          Common::CHECKER,
          Common::INVALID_PROTOBUF,
          ONNX_NAMESPACE::MakeString(
              "Function domain (",
              func_domain,
              ") does not exist. It should be pre-defined.")));
      return *init_status_;
    }

    auto lower_bound_incl = ver_range_it->second.first;
    auto upper_bound_incl = ver_range_it->second.second;
    if (!(lower_bound_incl <= func_version &&
          upper_bound_incl >= func_version)) {
      init_status_.reset(new Status(
          Common::CHECKER,
          Common::INVALID_PROTOBUF,
          ONNX_NAMESPACE::MakeString(
              "Function (",
              func_name,
              ") to be registered, but its version is not in the range of (",
              lower_bound_incl,
              ",",
              upper_bound_incl,
              ")")));
      return *init_status_;
    }

    if (function_set_[func_name][func_domain].count(func_version) > 0) {
      init_status_.reset(new Status(
          Common::CHECKER,
          Common::INVALID_PROTOBUF,
          ONNX_NAMESPACE::MakeString(
              "A function (",
              func_name,
              ") with version (",
              func_version,
              ") and domain (",
              func_domain,
              ") has already been registered.")));
      return *init_status_;
    }

    function_set_[func_name][func_domain][func_version] = function_proto;
  }

  return *init_status_;
}

FunctionBuilderRegistry::FunctionBuilderRegistry(
    const std::unordered_map<std::string, std::pair<int, int>>&
        domain_version_range) {
  domain_version_range_ = domain_version_range;
}

FunctionBuilderRegistry& FunctionBuilderRegistry::OnnxInstance() {
  static FunctionBuilderRegistry func_builder_registry(
      OpSchemaRegistry::DomainToVersionRange::Instance().Map());
  return func_builder_registry;
}

} // namespace ONNX_NAMESPACE
