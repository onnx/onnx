// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Resize in default domain from version 18 to 17

#pragma once

#include <memory>

#include "onnx/common/assertions.h"
#include "onnx/common/interned_strings.h"
#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Resize_18_17 final : public Adapter {
 public:
  Resize_18_17() : Adapter("Resize", OpSetID(18), OpSetID(17)) {}

  void adapt_resize_18_17(const std::shared_ptr<Graph>& /*unused*/, Node* node) const {
    const Symbol antialias("antialias");
    const Symbol coordinate_transformation_mode("coordinate_transformation_mode");
    const Symbol keep_aspect_ratio_policy("keep_aspect_ratio_policy");

    // Reject axes without rank information.
    ONNX_ASSERTM(
        !node->hasAttribute(kaxes),
        "Resize axes is not supported when converting from opset 18 "
        "to opset 17.");

    // half_pixel_symmetric was introduced in opset 19.
    if (node->hasAttribute(coordinate_transformation_mode)) {
      const auto& value = node->s(coordinate_transformation_mode);
      ONNX_ASSERTM(
          value != "half_pixel_symmetric",
          "Resize coordinate_transformation_mode='",
          value,
          "' is not supported when converting to opset 17.");
    }

    // The default antialias behavior is supported by opset 17.
    if (node->hasAttribute(antialias)) {
      const auto value = node->i(antialias);
      ONNX_ASSERTM(
          value == 0,
          "Resize antialias=",
          value,
          " is not supported when "
          "converting from opset 18 to opset 17.");
    }

    // Require "stretch" for a safe downgrade.
    if (node->hasAttribute(keep_aspect_ratio_policy)) {
      ONNX_ASSERTM(
          node->s(keep_aspect_ratio_policy) == "stretch",
          "Resize keep_aspect_ratio_policy must be 'stretch' when "
          "converting from opset 18 to opset 17.");
    }

    // Remove explicit defaults that have equivalent behavior.
    if (node->hasAttribute(antialias)) {
      node->removeAttribute(antialias);
    }
    if (node->hasAttribute(keep_aspect_ratio_policy)) {
      node->removeAttribute(keep_aspect_ratio_policy);
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_resize_18_17(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
