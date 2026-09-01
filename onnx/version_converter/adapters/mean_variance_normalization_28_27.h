// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for MeanVarianceNormalization in default domain from version 28 to 27

#pragma once

#include <memory>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {

class MeanVarianceNormalization_28_27 final : public Adapter {
 public:
  explicit MeanVarianceNormalization_28_27() : Adapter("MeanVarianceNormalization", OpSetID(28), OpSetID(27)) {}

  Node* adapt(std::shared_ptr<Graph> /*graph*/, Node* node) const override {
    if (node->hasAttribute(kepsilon)) {
      ONNX_ASSERTM(
          node->f(kepsilon) == 1e-9f,
          "MeanVarianceNormalization 28->27 downgrade: epsilon must equal the opset-28 default (1e-9), got ",
          node->f(kepsilon),
          ". A custom epsilon is not representable in opset 27.");
      node->removeAttribute(kepsilon);
    }
    return node;
  }
};

} // namespace onnx::version_conversion
