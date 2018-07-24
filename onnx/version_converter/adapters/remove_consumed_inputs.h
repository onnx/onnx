// Adapter for all ops that remove consumed_inputs

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct RemoveConsumedInputs final : public Adapter {
  explicit RemoveConsumedInputs(const std::string op_name, const OpSetID
    initial, const OpSetID target): Adapter(std::move(op_name), std::move(
      initial), std::move(target)) {}

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    if (node->hasAttribute(kconsumed_inputs)) node->removeAttribute(kconsumed_inputs);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
