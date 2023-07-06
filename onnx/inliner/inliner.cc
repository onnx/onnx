// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "onnx/common/assertions.h"
#include "onnx/common/constants.h"
#include "onnx/common/interned_strings.h"
#include "onnx/inliner/inliner.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"

namespace ONNX_NAMESPACE {
namespace inliner {

namespace { // internal/private API

// Attribute lookup function. Returns nullptr if attribute is not found.
using AttributeLookupFunction = std::function<const AttributeProto*(const std::string& name)>;

using AttributeMap = std::unordered_map<std::string, const AttributeProto*>;

// Function lookup function. Returns true iff lookup is successful, in which case
// the found FunctionProto is copied into *return_value. The opset version is not
// a parameter since it is determined by the domain for a given model.
using FunctionResolver =
    std::function<bool(const std::string& domain, const std::string& op, FunctionProto* return_value)>;

// We use a string of the form "domain::name" as the id for a function.
using FunctionId = std::string;

FunctionId GetFunctionId(const std::string& domain, const std::string& op) {
  return NormalizeDomain(domain) + "::" + op;
}

FunctionId GetFunctionId(const FunctionProto& function) {
  return GetFunctionId(function.domain(), function.name());
}

FunctionId GetCalleeId(const NodeProto& node) {
  return GetFunctionId(node.domain(), node.op_type());
}

using OpsetMapBase = std::unordered_map<std::string, int64_t>;

struct OpsetMap : public OpsetMapBase {
  OpsetMap(const google::protobuf::RepeatedPtrField<OperatorSetIdProto>& list) {
    for (const auto& pair : list) {
      (*this)[NormalizeDomain(pair.domain())] = pair.version();
    }
  }

  OpsetMapBase Mismatches(const google::protobuf::RepeatedPtrField<OperatorSetIdProto>& list) {
    OpsetMapBase result;
    for (const auto& pair : list) {
      auto iter = this->find(NormalizeDomain(pair.domain()));
      if ((iter != this->end()) && (iter->second != pair.version()))
        result.insert(*iter);
    }
    return result;
  }
};

using RepeatedNodeProto = google::protobuf::RepeatedPtrField<NodeProto>;

// A visitor class for visiting all nodes and subgraphs in a graph.
struct NodeVisitorBase {
  void VisitGraph(const GraphProto& graph) {
    if (ProcessGraph(graph))
      for (auto& node : graph.node())
        VisitNode(node);
  }

  void VisitNode(const NodeProto& node) {
    if (ProcessNode(node)) {
      for (auto& attr : node.attribute()) {
        if (attr.has_g()) {
          VisitGraph(attr.g());
        }
        for (auto& graph : attr.graphs())
          VisitGraph(graph);
      }
    }
  }

  virtual bool ProcessGraph(const GraphProto& graph) {
    ONNX_UNUSED_PARAMETER(graph);
    return true;
  }

  virtual bool ProcessNode(const NodeProto& node) {
    ONNX_UNUSED_PARAMETER(node);
    return true;
  }

  virtual ~NodeVisitorBase() {}
};

struct NodeVisitor : public NodeVisitorBase {
  std::function<bool(const NodeProto&)> process_node;

  NodeVisitor(std::function<bool(const NodeProto&)> process_node_fun) : process_node(process_node_fun) {}

  bool ProcessNode(const NodeProto& node) override {
    return process_node(node);
  }
};

class NameGenerator : public NodeVisitorBase {
 public:
  NameGenerator(const GraphProto& graph) : index_(0) {
    VisitGraph(graph);
  }

  // Creates a new unique name, based on a suggested name, and adds it to the set
  // of existing names. Returns the newly created name.
  std::string CreateNew(const std::string& suggested) {
    std::string name = suggested;
    while (existing_names_.count(name) > 0) {
      name = suggested + "_" + std::to_string(index_++);
    }
    existing_names_.insert(name);
    return name;
  }

  void Add(const std::string& name) {
    // We don't bother to check for empty string names. Ok to add them.
    existing_names_.insert(name);
  }

  bool ProcessGraph(const GraphProto& graph) override {
    for (const auto& x : graph.input())
      Add(x.name());
    for (const auto& x : graph.initializer())
      Add(x.name());
    // Adding graph outputs is redundant for a valid graph, but we do it anyway,
    // to produce better results for invalid graphs.
    for (const auto& x : graph.output())
      Add(x.name());
    return true;
  }

  bool ProcessNode(const NodeProto& node) override {
    // We use a single name-space for node names and variable names, to keep name-generation simple.
    Add(node.name());
    for (const std::string& name : node.input()) {
      Add(name);
    }
    for (const std::string& name : node.output()) {
      Add(name);
    }
    return true;
  }

 private:
  unsigned int index_;
  std::unordered_set<std::string> existing_names_;
};

class Specializer {
 private:
  std::string suffix;
  NameGenerator& generator;
  AttributeLookupFunction attr_map;
  std::vector<std::unordered_map<std::string, std::string>> rename_scopes;

  Specializer(std::string suffix_, NameGenerator& generator_, AttributeLookupFunction attr_map_)
      : suffix(suffix_), generator(generator_), attr_map(attr_map_) {
    // Create an empty mapping for the top-level scope.
    rename_scopes.emplace_back();
  }

  // We use a two-level renaming scheme to generate names for variables when inlined in the
  // main graph. First, we add a suffix (specific to the call-site being inlined).
  // Thus, "temp" in called-function becomes "temp__1" for the first inlined function-call
  // and "temp__2" for the second inlined function-call. In addition, there is a subsequent
  // iterative check that ensures that this names does not clash with any pre-existing names,
  // and tries another counter-based suffix in the case of a clash, stopping when successful.
  std::string MakeUnique(const std::string& name) {
    return generator.CreateNew(name + suffix);
  }

  // Replace given name with a unique version of the name, and cache the
  // renaming-binding in current scope.
  void Rename(std::string& name) {
    auto new_name = MakeUnique(name);
    auto& current_scope = rename_scopes.back();
    current_scope[name] = new_name;
    name = new_name;
  }

  void LookupOrRename(std::string& name, bool is_new_def) {
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
      Rename(name);
    }
    // Otherwise, it is a reference to an outer-scope variable that should not be renamed.
  }

  template <bool isOutput>
  void Bind(
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
      if (isOutput)
        if (rename_as.empty())
          rename_as = MakeUnique(formal);
      current_scope[formal] = rename_as;
      if (!rename_as.empty())
        formal = rename_as;
    }
    for (; i < formals.size(); ++i) {
      std::string& formal = *formals.Mutable(i);
      std::string rename_as = isOutput ? MakeUnique(formal) : std::string("");
      current_scope[formal] = rename_as;
      if (!rename_as.empty())
        formal = rename_as;
    }
  }

  // Process a node:
  void Transform(NodeProto& n) {
    if (!n.name().empty())
      n.set_name(MakeUnique(n.name()));

    for (auto& x : *n.mutable_input()) {
      LookupOrRename(x, false);
    }
    for (auto& y : *n.mutable_output()) {
      LookupOrRename(y, true);
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
        Transform(*attr.mutable_g());
      }
      for (auto& graph : *attr.mutable_graphs())
        Transform(graph);
      ++attr_iter;
    }
  }

  // Process a sub-graph, contained as an attribute in a control-flow op node.
  void Transform(GraphProto& graph) {
    rename_scopes.emplace_back();
    for (auto& x : *graph.mutable_input())
      Rename(*x.mutable_name());
    for (auto& init : *graph.mutable_initializer())
      Rename(*init.mutable_name());
    for (auto& y : *graph.mutable_output())
      Rename(*y.mutable_name());
    for (auto& n : *graph.mutable_node())
      Transform(n);
    rename_scopes.pop_back();
  }

 public:
  // The main specialization method: specialize a FunctionProto for a particular call-site.
  static void
  Specialize(const NodeProto& callnode, FunctionProto& callee, std::string unique_suffix, NameGenerator& generator) {
    AttributeMap map;
    for (auto& attr : callnode.attribute()) {
      map[attr.name()] = &attr;
    }
    auto lookup = [&](const std::string& name) -> const AttributeProto* {
      auto iter = map.find(name);
      return (iter != map.end()) ? iter->second : nullptr;
    };
    Specializer specializer(unique_suffix, generator, lookup);

    specializer.Bind<false>(*callee.mutable_input(), callnode.input());
    specializer.Bind<true>(*callee.mutable_output(), callnode.output());

    for (auto& n : *callee.mutable_node())
      specializer.Transform(n);
  }
};

// Identify the set of all "input" variables used by a given node.
// This includes the variables listed as node.input, as well as
// implicit inputs referred to in any graph-valued-attribute of the node.

struct ComputeUsedVars : public NodeVisitorBase {
  std::vector<std::string> result;

  bool ProcessNode(const NodeProto& node) override {
    for (auto& var : node.input()) {
      if (!var.empty()) {
        result.push_back(var);
      }
    }
    return true; // process sub-graphs
  }

  ComputeUsedVars(const NodeProto& node) {
    result.reserve(node.input_size());
    VisitNode(node);
  }
};

std::vector<std::string> GetUsedVars(const NodeProto& node) {
  return ComputeUsedVars(node).result;
}

using ConstNodeMap = std::unordered_map<std::string, const NodeProto*>;

ConstNodeMap FindConstantNodes(const GraphProto& graph) {
  ConstNodeMap result;
  for (const NodeProto& node : graph.node()) {
    if (IsOnnxDomain(node.domain()) && (node.op_type() == "Constant")) {
      result[node.output(0)] = &node;
    }
  }
  return result;
}

const TypeProto& GetType(const ModelProto& model, std::string var) {
  for (auto& vi : model.graph().value_info()) {
    if (vi.name() == var)
      return vi.type();
  }
  for (auto& vi : model.graph().input()) {
    if (vi.name() == var)
      return vi.type();
  }
  for (auto& vi : model.graph().output()) {
    if (vi.name() == var)
      return vi.type();
  }
  ONNX_ASSERTM(false, "Type unknown for %s", var.c_str());
}

void ConvertVersion(ModelProto& model, const NodeProto& call_node, FunctionProto& function, int target_version) {
  shape_inference::InferShapes(model);

  ModelProto function_as_model;
  function_as_model.set_ir_version(model.ir_version());
  *function_as_model.mutable_opset_import() = function.opset_import();

  GraphProto& graph = *function_as_model.mutable_graph();
  // The graph's inputs are all the variables used in the call_node.
  auto used_vars = GetUsedVars(call_node);
  auto constant_node_map = FindConstantNodes(model.graph());

  RepeatedNodeProto& function_nodes = *function.mutable_node();
  RepeatedNodeProto& nodes = *graph.mutable_node();
  nodes.Reserve(function_nodes.size() + used_vars.size());

  auto* inputs = graph.mutable_input();
  for (const auto& var : used_vars) {
    auto* new_input = inputs->Add();
    new_input->set_name(var);
    *new_input->mutable_type() = GetType(model, var);
    // Create a copy of constants used by the call_node.
    // We do not handle initializers-as-constants for now.
    auto it = constant_node_map.find(var);
    if (it != constant_node_map.end()) {
      *nodes.Add() = *(it->second);
    }
  }

  // outputs: from call_node node outputs
  auto* outputs = graph.mutable_output();
  for (const auto& var : call_node.output()) {
    if (!var.empty()) {
      auto* new_output = outputs->Add();
      new_output->set_name(var);
      *new_output->mutable_type() = GetType(model, var);
    }
  }

  // TODO: Use std::move when it is fully supported on all protobuf platforms used
  for (auto& function_node : function_nodes)
    *nodes.Add() = function_node;
  function_nodes.Clear();

  auto converted = ONNX_NAMESPACE::version_conversion::ConvertVersion(function_as_model, target_version);

  function_nodes.Swap(converted.mutable_graph()->mutable_node());

  // Append new initializers to main graph initializers
  for (auto& added_initializer : converted.graph().initializer())
    *model.mutable_graph()->mutable_initializer()->Add() = added_initializer;
  for (auto& added_initializer : converted.graph().sparse_initializer())
    *model.mutable_graph()->mutable_sparse_initializer()->Add() = added_initializer;
}

constexpr int64_t kNoConversion = -1;
using FunctionMap = std::unordered_map<FunctionId, std::pair<const FunctionProto*, int64_t>>;

void InlineFunctions(ModelProto& model, FunctionMap map, bool convert_version) {
  auto* graph = model.mutable_graph();

  NameGenerator name_generator(*graph);

  auto* nodes = graph->mutable_node();
  google::protobuf::RepeatedPtrField<NodeProto> original_nodes;
  // Move all nodes into original_nodes
  original_nodes.Swap(nodes);

  int inline_count = 0;
  std::function<void(NodeProto & node)> append_node = [&](NodeProto& node) {
    FunctionProto callee;
    auto iter = map.find(GetCalleeId(node));
    if (iter != map.end()) {
      callee = *iter->second.first;
      int64_t target_version = iter->second.second;
      // Rename and specialize called function body
      Specializer::Specialize(node, callee, "__" + std::to_string(++inline_count), name_generator);
      if (target_version != kNoConversion) {
        ONNX_ASSERTM(
            convert_version,
            "Internal Error: Inlining function %s::%s requires version conversion, but convert_version is false.",
            callee.domain().c_str(),
            callee.name().c_str());
        ConvertVersion(model, node, callee, target_version);
      }
      // Append nodes of called function
      for (auto& callee_node : *callee.mutable_node())
        append_node(callee_node);
    } else {
      // Append node without inlining.
      // TODO: use std::move instead of copying. Use of move doesn't seem to work with
      // protobuf in some platforms/settings. [nodes->Add(std::move(node));]
      *nodes->Add() = node;
    }
  };
  for (auto& node : original_nodes) {
    append_node(node);
  }
}

} // namespace

// Public API implementation:

void InlineLocalFunctions(ModelProto& model, bool convert_version) {
  OpsetMap model_imports(model.opset_import());
  FunctionMap map;

  // For every function, we check if there is a mismatch between the opset versions
  // required for the function and the model. If there is no mismatch, we can inline
  // this function. If there is a mismatch only for the standard ONNX domain, we
  // can inline after version-conversion (if the version-conversion is successful).
  // Otherwise, we cannot inline, since currently version-conversion supports only
  // standard ONNX domain.

  for (auto& function : model.functions()) {
    auto mismatches = model_imports.Mismatches(function.opset_import());
    auto iter = mismatches.find(ONNX_DOMAIN);
    int64_t target_onnx_version = kNoConversion;
    if (convert_version && (iter != mismatches.end())) {
      target_onnx_version = iter->second;
      mismatches.erase(iter);
    }
    if (mismatches.empty()) {
      map[GetFunctionId(function)] = std::pair<const FunctionProto*, int64_t>(&function, target_onnx_version);
    }
  }

  InlineFunctions(model, map, convert_version);

  // Remove all model-local functions. We do not remove functions with a mis-matched
  // opset version. They need to be handled some other way, eg., using a version-adapter.
  auto* local_functions = model.mutable_functions();
  for (auto it = local_functions->begin(); it != local_functions->end();) {
    if (map.count(GetFunctionId(*it)) > 0)
      it = local_functions->erase(it);
    else
      ++it;
  }
}

} // namespace inliner
} // namespace ONNX_NAMESPACE
