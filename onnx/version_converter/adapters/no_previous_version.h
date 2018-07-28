// Adapter indicating lack of a previous version of some op before a given
// opset version.

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct NoPreviousVersionAdapter final : public Adapter {
  explicit NoPreviousVersionAdapter(const std::string& op_name, const OpSetID
    initial, const OpSetID target): Adapter(op_name,
    std::move(initial), std::move(target)) {}

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    ONNX_ASSERTM(false, "No Previous Version of %s exists", name().c_str());
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
