// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <functional>

#include "onnx/common/assertions.h"
#include "onnx/inliner/inliner.h"

namespace ONNX_NAMESPACE {
namespace inliner {

// Attribute lookup function. Returns nullptr if attribute is not found.
using AttributeLookupFunction = std::function<const AttributeProto*(const std::string& name)>;

using AttributeMap = std::unordered_map<std::string, const AttributeProto*>;

// Function lookup function. Returns true iff lookup is successful, in which case
// the found FunctionProto is copied into *return_value. The opset version is not
// a parameter since it is determined by the domain for a given model.
using FunctionResolver =
    std::function<bool(const std::string& domain, const std::string& op, FunctionProto* return_value)>;

using FunctionMap = std::unordered_map<std::string, const FunctionProto*>;

class Specializer {
 private:
  std::string prefix;
  AttributeLookupFunction attr_map;
  std::vector<std::unordered_map<std::string, std::string>> rename_scopes;

  Specializer(std::string prefix_, AttributeLookupFunction attr_map_) : prefix(prefix_), attr_map(attr_map_) {
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
    if (name.empty())
      return;
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
  void bind(
      google::protobuf::RepeatedPtrField<std::string>& formals,
      const google::protobuf::RepeatedPtrField<std::string>& actuals) {
    // Every formal parameter name FP should be replace by the corresponding actual parameter name AP.
    // However, if AP is empty, it is a missing optional parameter. This does not make any difference
    // for inputs. However, for outputs we use a unique dummy name to handle the case that it
    // is used in an output-context where it is not optional.
    ONNX_ASSERTM(
        actuals.size() <= formals.size(), "Number of actual parameters cannot exceed number of formal parameters");
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
        auto* attr_val = attr_map(attr.ref_attr_name());
        if (attr_val != nullptr) {
          // Copy value of attribute, but retain original name:
          std::string name = attr.name();
          attr = *attr_val;
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
  static void specialize(const NodeProto& callnode, FunctionProto& callee, std::string unique_prefix) {
    AttributeMap map;
    for (auto& attr : callnode.attribute()) {
      map[attr.name()] = &attr;
    }
    auto lookup = [&](const std::string& name) -> const AttributeProto* {
      auto iter = map.find(name);
      return (iter != map.end()) ? iter->second : nullptr;
    };
    Specializer specializer(unique_prefix, lookup);

    specializer.bind<false>(*callee.mutable_input(), callnode.input());
    specializer.bind<true>(*callee.mutable_output(), callnode.output());

    for (auto& n : *callee.mutable_node())
      specializer.transform(n);
  }
};

void inline_functions(ModelProto& model, FunctionResolver resolver) {
  auto* graph = model.mutable_graph();
  auto* nodes = graph->mutable_node();
  google::protobuf::RepeatedPtrField<NodeProto> original_nodes;
  // Move all nodes into original_nodes
  original_nodes.Swap(nodes);
  std::function<void(NodeProto & node)> append_node = [&](NodeProto& node) {
    FunctionProto callee;
    if (resolver(node.domain(), node.op_type(), &callee)) {
      // Rename and specialize called function body
      Specializer::specialize(node, callee, "aha");
      // Append nodes of called function
      for (auto& callee_node : *callee.mutable_node())
        append_node(callee_node);
    } else {
      // Append node without inlining.
      nodes->Add(std::move(node));
    }
  };
  for (auto& node : original_nodes) {
    append_node(node);
  }
}

void inline_local_functions(ModelProto& model) {
  FunctionMap map;

  for (auto& function : model.functions()) {
    auto name = function.domain() + "::" + function.name();
    map[name] = &function;
  }

  auto get_function = [&](const std::string& domain, const std::string& op, FunctionProto* return_value) -> bool {
    auto iter = map.find(domain + "::" + op);
    if (iter != map.end()) {
      *return_value = *iter->second;
      return true;
    }
    return false;
  };

  inline_functions(model, get_function);
  model.clear_functions();
}

} // namespace inliner
} // namespace ONNX_NAMESPACE
