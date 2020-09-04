// Adapter for RNN in default domain from version 13 to 12

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct RNN_13_12 final : public Adapter {
  explicit RNN_13_12()
    : Adapter("RNN", OpSetID(13), OpSetID(12)) {
    }

  void adapt_rnn_13_12(std::shared_ptr<Graph> graph, Node* node) const {
      // Remove the batch_major attribute
      if (node->hasAttribute(kbatch_major)) {
        ONNX_ASSERTM(node->i(kbatch_major) == 0, "RNN in Opset "
            "Version 12 does not support batch major.");
        node->removeAttribute(kbatch_major);
      }
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_rnn_13_12(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion