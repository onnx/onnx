// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/defs/function.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
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

std::vector<NodeProto> FunctionBodyHelper::BuildNodes(
    const std::vector<NodeDef>& node_defs) {
  std::vector<NodeProto> nodes(node_defs.size());

  for (size_t i = 0; i < node_defs.size(); i++) {
    const NodeDef& node = node_defs[i];
    NodeProto& n = nodes[i];

    n.set_op_type(node.op_type);
    for (const auto& i : node.inputs) {
      n.add_input(i);
    }
    for (const auto& o : node.outputs) {
      n.add_output(o);
    }
    for (const auto& attr : node.attributes) {
      *(n.add_attribute()) = attr.proto;
    }
  }

  return nodes;
}

} // namespace ONNX_NAMESPACE
