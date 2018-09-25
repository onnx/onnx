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

Common::Status FunctionBuilderRegistry::Register(
    const FunctionBuilder& function_builder) {
  std::lock_guard<std::mutex> lock(mutex_);
  function_builders.push_back(function_builder);
  return Common::Status::OK();
}

// Get functions for specific domain.
Common::Status FunctionBuilderRegistry::GetFunctions(
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
    op_set.insert(
        {func_builder.GetDomain(), (int)function_proto->since_version()});
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

std::unique_ptr<FunctionProto> FunctionBuilderRegistry::GetFunction(
    const std::string& func_name,
    const int maxInclusiveVersion,
    const std::string& domain) const {
  std::multimap<std::string, std::unique_ptr<FunctionProto>> funcs;
  auto status = GetFunctions(domain, &funcs);
  if (!status.IsOK()) {
    return nullptr;
  }
  std::map<int, std::unique_ptr<FunctionProto>> version_to_func;
  auto range = funcs.equal_range(func_name);
  for (auto i = range.first; i != range.second; ++i) {
    version_to_func[static_cast<int>(i->second->since_version())] =
        std::move(i->second);
  }

  if (version_to_func.empty()) {
    return nullptr;
  }
  auto pos = version_to_func.lower_bound(maxInclusiveVersion);
  if (version_to_func.begin() == pos && pos->first > maxInclusiveVersion) {
    return nullptr;
  }
  if (version_to_func.end() == pos || pos->first > maxInclusiveVersion) {
    // All versions are less than specified version, or,
    // The <pos> version is greater than specified version.
    pos--;
  }
  return std::move(pos->second);
}

FunctionBuilderRegistry& FunctionBuilderRegistry::OnnxInstance() {
  static FunctionBuilderRegistry func_builder_registry;
  return func_builder_registry;
}

std::string InteralTensorNameGenerator(
    const std::string& node_name,
    const std::string& internal_name) {
  std::string new_name = "Func_" + node_name + internal_name;
  return new_name;
}

void FunctionExpandHelper(
    const NodeProto& node,
    const FunctionProto& func,
    GraphProto& g,
    const std::string& node_prefix) {
  // Create a temporary unique node prefix for tensor names
  std::string uniq_prefix = node_prefix;
  if (uniq_prefix.empty()) {
    const void* address = static_cast<const void*>(&node);
    std::stringstream ss;
    ss << address;
    uniq_prefix = ss.str();
  }
  std::string node_name =
      node.has_name() ? node.name() : func.name() + uniq_prefix;
  int version = (int)func.since_version();
  std::unordered_map<std::string, std::string> input_names_map;
  std::unordered_map<std::string, std::string> output_names_map;
  std::unordered_map<std::string, AttributeProto> attr_map;

  for (int idx = 0; idx < node.input_size(); ++idx) {
    if (idx >= func.input_size()) {
      throw std::runtime_error(
          "Input for function node " + node_name + " is out of bounds");
    }
    input_names_map[func.input().Get(idx)] = node.input().Get(idx);
  }
  for (int idx = 0; idx < node.output_size(); ++idx) {
    if (idx >= func.output_size()) {
      throw std::runtime_error(
          "Output for function node " + node_name + " is out of bounds");
    }
    output_names_map[func.output().Get(idx)] = node.output().Get(idx);
  }

  for (auto& attr : node.attribute()) {
    attr_map[attr.name()] = attr;
  }

  for (auto& function_node : func.node()) {
    NodeProto* new_node = g.add_node();
    new_node->CopyFrom(function_node);
    new_node->clear_input();
    new_node->clear_output();
    new_node->clear_attribute();
    for (auto& input : function_node.input()) {
      if (input_names_map.count(input)) {
        new_node->add_input(input_names_map[input]);
      } else {
        new_node->add_input(InteralTensorNameGenerator(node_name, input));
      }
    }
    for (auto& output : function_node.output()) {
      if (output_names_map.count(output)) {
        new_node->add_output(output_names_map[output]);
      } else {
        new_node->add_output(InteralTensorNameGenerator(node_name, output));
      }
    }
    for (auto& attr : function_node.attribute()) {
      if (attr.has_ref_attr_name()) {
        if (attr_map.count(attr.ref_attr_name())) {
          AttributeProto* new_attr = new_node->add_attribute();
          new_attr->CopyFrom(attr_map[attr.ref_attr_name()]);
        }
      } else {
        AttributeProto* new_attr = new_node->add_attribute();
        new_attr->CopyFrom(attr);
      }
    }
  }
}
} // namespace ONNX_NAMESPACE
