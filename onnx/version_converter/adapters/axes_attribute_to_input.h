/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for all ops that remove consumed_inputs

#pragma once

#include "onnx/version_converter/adapters/adapter.h"
#include "onnx/defs/attr_proto_util.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class AxesAttributeToInput : public Adapter {
 public:
  explicit AxesAttributeToInput(const std::string& op_name, const OpSetID& initial, const OpSetID& target)
      : Adapter(op_name, initial, target) {}

  void attrToInput(GraphProto* graph_proto, NodeProto* node_proto, std::vector<int64_t>& axes) const {
    node_proto->attribute();
    const std::string constant_node_name("aaa");
    NodeProto* constant_node_proto = graph_proto->add_node();
    constant_node_proto->set_op_type("Constant");
    constant_node_proto->set_name(constant_node_name);
    TensorProto tensor_proto = ToTensor(axes);
    *constant_node_proto->add_attribute() = MakeAttribute("attr_name", tensor_proto);
    node_proto->add_input(constant_node_name);
  }

  NodeProto* adapt(GraphProto* graph_proto, NodeProto* node_proto) const override {
    auto it = std::find_if(node_proto->attribute().begin(), node_proto->attribute().end(), [](AttributeProto& attr) {
      return attr.name() == "axes";
    });
    if (it != node_proto->attribute().end()) {
      std::vector<int64_t> axes(it->ints().begin(), it->ints().end());
      attrToInput(graph_proto, node_proto, axes);
      std::remove_if(node_proto->mutable_attribute()->begin(), node_proto->mutable_attribute()->end(), [](AttributeProto& attr){
        attr.name() == "axes";});
    }
    return node_proto;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
