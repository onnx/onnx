// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {

class Gemm_29_28 final : public Adapter {
 public:
  explicit Gemm_29_28() : Adapter("Gemm", OpSetID(29), OpSetID(28)) {}

  Node* adapt(std::shared_ptr<Graph> /*unused*/, Node* node) const override {
    const auto& inputs = node->inputs();
    ONNX_ASSERTM(inputs.size() >= 2, "Gemm requires at least two inputs.");
    ONNX_ASSERTM(
        inputs[0]->has_sizes() && inputs[0]->sizes().size() == 2,
        "Gemm downgrade to opset 28 requires input A to have known rank 2.");
    return node;
  }
};

} // namespace ONNX_NAMESPACE::version_conversion
