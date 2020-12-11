// Adapter for RNN in default domain from version 14 to 13

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct RNN_14_13 final : public Adapter {
  explicit RNN_14_13()
    : Adapter("RNN", OpSetID(14), OpSetID(13)) {
    }

  void adapt_rnn_14_13(std::shared_ptr<Graph> graph, Node* node) const {
      // Remove the batch_major attribute
      if (node->hasAttribute(kbatch_major)) {
        ONNX_ASSERTM(node->i(kbatch_major) == 0, "RNN in Opset "
            "Version 13 does not support batch major.");
        node->removeAttribute(kbatch_major);
      }
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_rnn_14_13(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion