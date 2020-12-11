// Adapter for RNN in default domain from version 13 to 14

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct RNN_13_14 final : public Adapter {
  explicit RNN_13_14()
    : Adapter("RNN", OpSetID(13), OpSetID(14)) {
    }

  void adapt_rnn_13_14(std::shared_ptr<Graph> graph, Node* node) const {
      // Add the batch_major attribute
      node->i_(kbatch_major, 0);
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_rnn_13_14(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion