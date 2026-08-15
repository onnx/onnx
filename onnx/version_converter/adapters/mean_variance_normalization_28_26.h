// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for MeanVarianceNormalization in default domain from version 28 to 26

#pragma once

#include <memory>

#include "onnx/version_converter/adapters/adapter.h"

namespace onnx::version_conversion {

// Removes the epsilon attribute introduced in opset 28, since opset 26 does
// not define that attribute. When epsilon equals the opset-28 default (1e-9)
// the resulting opset-26 node is semantically equivalent; if a non-default
// epsilon was used the conversion is lossy and a warning comment is attached.
class MeanVarianceNormalization_28_26 final : public Adapter {
 public:
  explicit MeanVarianceNormalization_28_26() : Adapter("MeanVarianceNormalization", OpSetID(28), OpSetID(26)) {}

  Node* adapt(std::shared_ptr<Graph> /*graph*/, Node* node) const override {
    if (node->hasAttribute(kepsilon)) {
      node->removeAttribute(kepsilon);
    }
    return node;
  }
};

} // namespace onnx::version_conversion
