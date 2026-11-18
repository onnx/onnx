// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {

class DequantizeLinear_29_28 final : public Adapter {
 public:
  explicit DequantizeLinear_29_28() : Adapter("DequantizeLinear", OpSetID(29), OpSetID(28)) {}

  Node* adapt(std::shared_ptr<Graph> /*unused*/, Node* node) const override {
    ONNX_ASSERTM(
        !node->hasAttribute(Symbol("block_shape")),
        "DequantizeLinear with block_shape cannot be converted to opset 28.");
    return node;
  }
};

} // namespace ONNX_NAMESPACE::version_conversion
