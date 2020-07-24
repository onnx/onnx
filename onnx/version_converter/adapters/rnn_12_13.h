// Adapter for RNN in default domain from version 12 to 13

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct RNN_12_13 final : public Adapter {
  explicit RNN_12_13()
    : Adapter("RNN", OpSetID(12), OpSetID(13)) {
    }

  void adapt_rnn_12_13(std::shared_ptr<Graph> graph, Node* node) const {
      // Add the time_major attribute
      node->i_(ktime_major, 1);
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_rnn_12_13(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion