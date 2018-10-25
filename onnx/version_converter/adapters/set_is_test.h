// Adapter for BatchNormalization and Dropout in default domain from version 7 to 6

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct SetIsTest final : public Adapter {
  explicit SetIsTest(const std::string& op_name, const OpSetID&
    initial, const OpSetID& target): Adapter(op_name, initial, target) {}

  void adapt_set_is_test(std::shared_ptr<Graph>, Node* node) const {
    node->i_(kis_test, 1);
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_set_is_test(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
