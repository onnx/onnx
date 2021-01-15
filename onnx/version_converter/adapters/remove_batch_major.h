// Adapter for GRU/LSTM/RNN in default domain from version 14 to 13

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct RemoveBatchMajor final : public Adapter {
  explicit RemoveBatchMajor(const std::string& op_name)
    : Adapter(op_name, OpSetID(14), OpSetID(13)) {
    }

  void adapt_remove_batch_major(Node* node) const {
      // Remove the batch_major attribute
      if (node->hasAttribute(kbatch_major)) {
        ONNX_ASSERTM(node->i(kbatch_major) == 0, "GRU/LSTM/RNN in Opset "
            "Version 13 does not support batch major.");
        node->removeAttribute(kbatch_major);
      }
  }

  void adapt(std::shared_ptr<Graph> , Node* node) const override {
    adapt_remove_batch_major(node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion