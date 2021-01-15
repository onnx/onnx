// Adapter for GRU/LSTM/RNN in default domain from version 13 to 14

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct AddBatchMajor final : public Adapter {
  explicit AddBatchMajor(const std::string& op_name)
    : Adapter(op_name, OpSetID(13), OpSetID(14)) {
    }

  void adapt_add_batch_major(Node* node) const {
      // Add the batch_major attribute
      node->i_(kbatch_major, 0);
  }

  void adapt(std::shared_ptr<Graph> , Node* node) const override {
    adapt_add_batch_major(node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion