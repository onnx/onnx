// Adapter indicating backwards compatibility of op between opsets with
// separate definitions

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct BackwardsCompatibleAdapter final : public Adapter {
  // TODO: Should these be &&?
  explicit BackwardsCompatibleAdapter(const std::string op_name, const OpSetID
    initial, const OpSetID target): Adapter(std::move(op_name), std::move(
      initial), std::move(target)) {}

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {}
};

}} // namespace ONNX_NAMESPACE::version_conversion
