/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/function_inline.h"
#include "onnx/common/file_utils.h"

namespace ONNX_NAMESPACE {
namespace function_inline {
using NameList = google::protobuf::RepeatedPtrField<std::string>;

// when a function is inlined, its nodes are copied to the main graph.
// to maintain connection and to avoid name clashing, we follow a name mapping:
// scope_name: outer scope_name + current_function_name
// input/output names of a node within function = scope-name + function-name + finction-encounter-count + original-input-or-output-name (+ UUID)
// an UUID is added in case there is still a name clashing.
// for example:
// scope name for function foo: "foo"
// input name for op ReduceMax within function foo: "foo" + "0" + "." + "x" = foo0.x
// the function's input and output names are updated in the same way.
struct SymbolContext {
  std::map<std::string, int> func_encounter_count_map_;
  std::map<const NodeProto*, std::string> nested_function_scope_name_map_;
};

std::string make_function_node_scope_name(
  const std::string& current_scope, const std::string& function_name, int function_encounter_count) {
    // current_scope: "foo"
    // function_encounter_count: 0
    // contained_node_function_name: foo_nested (foo2 is nested in foo)
    // return foo.0.foo_nested
    // encounter of a function means times that we have inlined a specific op_type of the function.
    if (current_scope.empty())
      // top level function node
      return function_name + "." + std::to_string(function_encounter_count);
    else if (function_encounter_count < 0)
      // nested function node just got copyed to main graph. at this point we do not know encounter count.
      return current_scope + "." + function_name;
    else if (function_name.empty())
      // nested function node that is to be inlinded
      return current_scope + "." + std::to_string(function_encounter_count);
    else
      return current_scope + "." + function_name + "." + std::to_string(function_encounter_count);
}

std::string make_node_io_name(
  const std::string& containing_function_scope_name, int containing_function_incounter_count, const std::string& origin_io_name) {
    return containing_function_scope_name + std::to_string(containing_function_incounter_count) + "." + origin_io_name;
}

int find_name(const NameList& name_list, const std::string& name) {
  for (int i = 0; i < name_list.size(); i++) {
    if (name_list[i] == name)
      return i;
  }
  return -1;
}

const FunctionProto* GetFunction(ModelProto& model, const std::string& function_name) {
  for (const FunctionProto& function_proto : model.functions()) {
      if (function_proto.name() == function_name)
        return &function_proto;
  }

  return nullptr;
}


void ModifyNodeSymbol(
  NodeProto& node,
  const NodeProto& containing_function_node,
  const FunctionProto& containing_function,
  const std::string& containing_function_node_scope_name,
  int containing_function_encounter_count) {
    for(int i = 0; i < node.input_size(); i++) {
      const std::string& input = node.input(i);
      int index = find_name(containing_function.input(), input);
      if (index >= 0) {
        node.set_input(i, containing_function_node.input(index));
      } else {
        node.set_input(i, make_node_io_name(containing_function_node_scope_name, containing_function_encounter_count, input));
      }
    }

    for(int i = 0; i < node.output_size(); i++) {
      const std::string& output = node.output(i);
      int index = find_name(containing_function.output(), output);
      if (index >= 0) {
        node.set_output(i, containing_function_node.output(index));
      } else {
        node.set_output(i, make_node_io_name(containing_function_node_scope_name, containing_function_encounter_count, output));
      }
    }
}

bool InlineFunction(ModelProto& model, GraphProto& graph, const NodeProto& node, SymbolContext& symbol_context) {
  // Remove the function node, add the nodes in function's subgraph into the main graph.
  const FunctionProto* function = GetFunction(model, node.op_type());
  if (!function)
    return false;

  if (symbol_context.func_encounter_count_map_.find(function->name()) == symbol_context.func_encounter_count_map_.end())
    symbol_context.func_encounter_count_map_[function->name()] = 0;
  else
    symbol_context.func_encounter_count_map_[function->name()]++;
  
  if (symbol_context.nested_function_scope_name_map_.find(&node) == symbol_context.nested_function_scope_name_map_.end())
    // this is for top level function nodes
    symbol_context.nested_function_scope_name_map_[&node] =
      make_function_node_scope_name("", function->name(), symbol_context.func_encounter_count_map_[function->name()]);
  else
    // this is for nested function nodes. its scope is already set in a previse scan where the node is copyed update with encounter count
    symbol_context.nested_function_scope_name_map_[&node] =
      make_function_node_scope_name(symbol_context.nested_function_scope_name_map_[&node], "", symbol_context.func_encounter_count_map_[function->name()]);

  for (int i = 0; i < function->node_size(); i++) {
    const NodeProto* current_child_node = &function->node(i);
    NodeProto* new_child_node = graph.add_node();
    new_child_node->CopyFrom(*current_child_node);
    ModifyNodeSymbol(
      *new_child_node,
      node,
      *function,
      symbol_context.nested_function_scope_name_map_[&node],
      symbol_context.func_encounter_count_map_[function->name()]);

    bool child_node_is_function = GetFunction(model, new_child_node->op_type()) != nullptr;
    if (child_node_is_function)
      // at this point we know parent node but not encounter count of 
      symbol_context.nested_function_scope_name_map_[new_child_node] = 
        make_function_node_scope_name(
          symbol_context.nested_function_scope_name_map_[&node],
          new_child_node->op_type(),
          -1);
  }
  return true;
}

void inline_model_function(ModelProto& model) {
  SymbolContext symbol_context;
  auto& graph = *model.mutable_graph();
  bool function_inlined = false;
  do {
    function_inlined = false;
    for (int i = 0; i < graph.node_size(); i++) {
      const NodeProto& node = graph.node(i);
      if (InlineFunction(model, graph, node, symbol_context)) {
        auto& mutable_node = *graph.mutable_node();
        mutable_node.DeleteSubrange(i, 1);
        function_inlined = true;
        break;
      }
    }
  } while (function_inlined);
}

void inline_model_function_path(const std::string& model_path, const std::string& target_model_path) {
  ModelProto model;
  LoadProtoFromPath(model_path, model);
  inline_model_function(model);
  SaveProto(&model, target_model_path);
}
} // namespace checker
} // namespace ONNX_NAMESPACE
