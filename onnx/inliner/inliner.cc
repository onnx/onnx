// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <functional>

#include "onnx/inliner/inliner.h"

namespace ONNX_NAMESPACE {
namespace Inliner {

// Attribute lookup function. Returns nullptr if attribute is not found.
using AttributeLookupFunction = std::function<const AttributeProto*(const std::string& name)>;

class Inliner {
 private:
  std::string prefix;
  AttributeLookupFunction attr_map;
  std::vector<std::unordered_map<std::string, std::string>> rename_scopes;

  Inliner(std::string prefix_, AttributeLookupFunction attr_map_) : prefix(prefix_),
                                                                               attr_map(attr_map_) {
    // Create an empty mapping for the top-level scope.
    rename_scopes.emplace_back();
  }

  // Replace given name with a unique version of the name, and cache the
  // renaming-binding in current scope.
  void make_unique(std::string& name) {
    auto new_name = prefix + name;
    auto& current_scope = rename_scopes.back();
    current_scope[name] = new_name;
    name = new_name;
  }

  void rename(std::string& name, bool is_new_def) {
    if (name.empty()) return;
    for (auto i = rename_scopes.size(); i > 0; --i) {
      const auto& map = rename_scopes[i - 1];
      auto iter = map.find(name);
      if (iter != map.end()) {
        name = iter->second;
        return;
      }
    }
    if (is_new_def) {
      make_unique(name);
    }
    // Otherwise, it is a reference to an outer-scope variable that should not be renamed.
  }

  template <bool isOutput>
  void bind(google::protobuf::RepeatedPtrField<string>& formals, const google::protobuf::RepeatedPtrField<string>& actuals) {
    // Every formal parameter name FP should be replace by the corresponding actual parameter name AP.
    // However, if AP is empty, it is a missing optional parameter. This does not make any difference
    // for inputs. However, for outputs we use a unique dummy name to handle the case that it
    // is used in an output-context where it is not optional.
    ORT_ENFORCE(actuals.size() <= formals.size(),
                "Number of actual parameters cannot exceed number of formal parameters");
    auto& current_scope = rename_scopes.back();
    int i = 0;
    for (; i < actuals.size(); ++i) {
      std::string& formal = *formals.Mutable(i);
      std::string rename_as = actuals.Get(i);
      if constexpr (isOutput)
        if (rename_as.empty())
          rename_as = prefix + formal;
      current_scope[formal] = rename_as;
      if (!rename_as.empty())
        formal = rename_as;
    }
    for (; i < formals.size(); ++i) {
      std::string& formal = *formals.Mutable(i);
      std::string rename_as = isOutput ? prefix + formal : std::string("");
      current_scope[formal] = rename_as;
      if (!rename_as.empty())
        formal = rename_as;
    }
  }

  // Process a node:
  void transform(NodeProto& n) {
    if (!n.name().empty())
      n.set_name(prefix + n.name());

    for (auto& x : *n.mutable_input()) {
      rename(x, false);
    }
    for (auto& y : *n.mutable_output()) {
      rename(y, true);
    }
    auto& attributes = *n.mutable_attribute();
    for (auto attr_iter = attributes.begin(); attr_iter != attributes.end();) {
      auto& attr = *attr_iter;
      if (!attr.ref_attr_name().empty()) {
        // Attribute-references must be replaced by the corresponding attribute-value in the call-node
        // if the call-node contains the attribute. Otherwise, this attribute must be removed.
        auto entry = attr_map.find(attr.ref_attr_name());
        if (entry != attr_map.cend()) {
          // Copy value of attribute, but retain original name:
          std::string name = attr.name();
          attr = entry->second;
          attr.set_name(name);
        } else {
          attr_iter = attributes.erase(attr_iter);
          continue;
        }
      }
      // Subgraphs must be recursively processed.
      if (attr.has_g()) {
        transform(*attr.mutable_g());
      }
      for (auto& graph : *attr.mutable_graphs())
        transform(graph);
      ++attr_iter;
    }
  }

  // Process a sub-graph, contained as an attribute in a control-flow op node.
  void transform(GraphProto& graph) {
    rename_scopes.emplace_back();
    for (auto& x : *graph.mutable_input())
      make_unique(*x.mutable_name());
    for (auto& init : *graph.mutable_initializer())
      make_unique(*init.mutable_name());
    for (auto& y : *graph.mutable_output())
      make_unique(*y.mutable_name());
    for (auto& n : *graph.mutable_node())
      transform(n);
    rename_scopes.pop_back();
  }

 public:
  // The main specialization method: specialize a FunctionProto for a particular call-site.
  static void specialize(const NodeProto& callnode, FunctionProto& callee, AttributeLookupFunction attr_map, std::string unique_prefix) {
    Inliner inliner(unique_prefix, attr_map);

    inliner.bind<false>(*callee.mutable_input(), callnode.input());
    inliner.bind<true>(*callee.mutable_output(), callnode.output());

    for (auto& n : *callee.mutable_node())
      inliner.transform(n);
  }
};

}  // namespace Inliner
}  // namespace ONNX_NAMESPACE
