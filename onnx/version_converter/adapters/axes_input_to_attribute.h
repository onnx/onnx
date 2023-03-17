/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for all ops that remove consumed_inputs

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class AxesInputToAttribute : public Adapter {
 public:
  explicit AxesInputToAttribute(const std::string& op_name, const OpSetID& initial, const OpSetID& target)
      : Adapter(op_name, initial, target) {}

  NodeProto* adapt(GraphProto* graph, NodeProto* node) const override {
    // Identify if axes is statically determined; if so, feed as attribute
    auto it_node = std::find_if(graph->node().begin(), graph->node().end(), [node](NodeProto& n) { return n.name() == node->input()[1]; });
    if (it_node != graph->node().end() && it_node->op_type() == "Constant") {
      const NodeProto& constant_node = *it_node;
      auto& axes_proto = constant_node.attribute()[0].ints();
      const std::vector<int64_t> axes(axes_proto.begin(), axes_proto.end());
      *node->add_attribute() = MakeAttribute("attr_name", axes);
    } else {
      auto it_initializer =
          std::find_if(graph->initializer().begin(), graph->initializer().end(), [node](TensorProto& t) {
            return t.name() == node->input()[1];
          });
      ONNX_ASSERTM(it_initializer != graph->initializer().end(), "No initializer or constant input to node found");
        const TensorProto& axes_proto = *it_initializer;
        *node->add_attribute() = MakeAttribute("attr_name", axes_proto);
        std::remove_if(graph->initializer().begin(), graph->initializer().end(), [node](TensorProto& t) {
          return t.name() == node->input()[1];
        });
    }
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
