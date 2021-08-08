/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for RoiAlign in default domain from version 15 to 16

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct RoiAlign_15_16 final: public Adapter {
  explicit RoiAlign_15_16() : Adapter("RoiAlign", OpSetID(15), OpSetID(16)) {}

  void adapt_roialign_15_16(std::shared_ptr<Graph>, Node* node) const {
    Symbol coordinate_transformation_mode = Symbol("coordinate_transformation_mode");
    node->s_(coordinate_transformation_mode, "output_half_pixel");
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_roialign_15_16(graph, node);
    return node;
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
