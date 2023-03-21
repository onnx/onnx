/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter indicating compatibility of op between opsets with separate
// definitions

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

struct CompatibleAdapter final : public Adapter {
  explicit CompatibleAdapter(const std::string& op_name, const OperatorSetIdProto& initial, const OperatorSetIdProto& target)
      : Adapter(op_name, initial, target) {}

  NodeProto* adapt(std::shared_ptr<GraphProto>, NodeProto* node) const override {
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
