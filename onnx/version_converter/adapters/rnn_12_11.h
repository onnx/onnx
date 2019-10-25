// Adapter for RNN in default domain from version 12 to 11

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct RNN_12_11 final : public Adapter {
  explicit RNN_12_11()
    : Adapter("RNN", OpSetID(12), OpSetID(11)) {
    }

  void adapt_rnn_12_11(std::shared_ptr<Graph> graph, Node* node) const {
      // Remove the time_major attribute
      if (node->hasAttribute(ktime_major)) {
        ONNX_ASSERTM(node->i(ktime_major) == 1, "RNN in Opset "
            "Version 10 does not support batch major.");
        node->removeAttribute(ktime_major);
      }
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_rnn_12_11(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion