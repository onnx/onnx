// Adapter for RNN in default domain from version 11 to 12

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct RNN_11_12 final : public Adapter {
  explicit RNN_11_12()
    : Adapter("RNN", OpSetID(11), OpSetID(12)) {
    }

  void adapt_rnn_11_12(std::shared_ptr<Graph> graph, Node* node) const {
      // Add the time_major attribute
      node->i_(ktime_major, 1);
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_rnn_11_12(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion