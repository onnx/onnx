/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for broadcasting ops in default domain from version 7 to 6

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class BroadcastBackwardCompatibility final : public Adapter {
 public:
  explicit BroadcastBackwardCompatibility(const std::string& op_name, const OpSetID& initial, const OpSetID& target)
      : Adapter(op_name, initial, target) {}

  void adapt_broadcast_backward_compatibility(GraphProto* graph_proto, NodeProto* node) const {
    // Verify that broadcasts are allowed in limited spec of opset version 6
    // Multidirectional broadcasting, as defined in Broadcasting.md
    // MathDocGenerator provides differences
    // Main change: encode broadcasting commands as explicit attribute
    auto it = std::find_if(graph_proto->value_info().begin(), graph_proto->value_info().end(), [node](ValueInfoProto& v) {
      return v.name() == node->input(0);
    });
    const ValueInfoProto& value_info0 = *it;
    it = std::find_if(graph_proto->value_info().begin(), graph_proto->value_info().end(), [node](ValueInfoProto& v) {
      return v.name() == node->input(1);
    });
    const ValueInfoProto& value_info1 = *it;
    const TypeProto& type_proto0 = value_info0.type();
    std::vector<TensorShapeProto_Dimension> dim1 =
        protobuf_repeated_fields_to_vector<TensorShapeProto_Dimension>(type_proto0.tensor_type().shape().dim());
    const TypeProto& type_proto1 = value_info1.type();
    std::vector<TensorShapeProto_Dimension> dim2 =
        protobuf_repeated_fields_to_vector<TensorShapeProto_Dimension>(type_proto1.tensor_type().shape().dim());
    int req_broadcast = check_numpy_unibroadcastable_and_require_broadcast(dim1, dim2);
    ONNX_ASSERTM(
        req_broadcast != -1,
        "%s being converted from %d to %d does "
        "not have broadcastable inputs.",
        name().c_str(),
        initial_version().version(),
        target_version().version());

    if (req_broadcast == 1) {
      // If conditional is not fulfilled, we have a default broadcast
      // Add broadcast attribute
      *node->add_attribute() = MakeAttribute("aa", (int64_t)1);
    }
  }

  NodeProto* adapt(GraphProto* graph, NodeProto* node) const override {
    adapt_broadcast_backward_compatibility(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
