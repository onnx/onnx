// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for MeanVarianceNormalization in default domain from version 29 to 28

#pragma once

#include <memory>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {

class MeanVarianceNormalization_29_28 final : public Adapter {
 public:
  explicit MeanVarianceNormalization_29_28() : Adapter("MeanVarianceNormalization", OpSetID(29), OpSetID(28)) {}

  Node* adapt(std::shared_ptr<Graph> /*graph*/, Node* node) const override {
    if (node->hasAttribute(kepsilon)) {
      ONNX_ASSERTM(
          node->f(kepsilon) == 1e-9f,
          "MeanVarianceNormalization 29->28 downgrade: epsilon must equal the opset-29 default (1e-9), got ",
          node->f(kepsilon),
          ". A custom epsilon is not representable in opset 28.");
      node->removeAttribute(kepsilon);
    }
    return node;
  }
};

} // namespace ONNX_NAMESPACE::version_conversion
