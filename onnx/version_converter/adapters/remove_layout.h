// Adapter for GRU/LSTM/RNN in default domain from version 14 to 13

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct RemoveLayout final : public Adapter {
  explicit RemoveLayout(const std::string& op_name)
    : Adapter(op_name, OpSetID(14), OpSetID(13)) {
    }

  void adapt_remove_layout(Node* node) const {
      // Remove the layout attribute
      if (node->hasAttribute(klayout)) {
        ONNX_ASSERTM(node->i(klayout) == 0, "GRU/LSTM/RNN in Opset "
            "Version 13 does not support layout.");
        node->removeAttribute(klayout);
      }
  }

  Node* adapt(std::shared_ptr<Graph> , Node* node) const override {
    adapt_remove_layout(node);
    return node;
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion