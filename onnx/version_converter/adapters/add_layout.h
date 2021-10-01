// Adapter for GRU/LSTM/RNN in default domain from version 13 to 14

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct AddLayout final : public Adapter {
  explicit AddLayout(const std::string& op_name)
    : Adapter(op_name, OpSetID(13), OpSetID(14)) {
    }

  void adapt_add_layout(Node* node) const {
      // Add the layout attribute
      node->i_(klayout, 0);
  }

  Node* adapt(std::shared_ptr<Graph> , Node* node) const override {
    adapt_add_layout(node);
    return node;
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion