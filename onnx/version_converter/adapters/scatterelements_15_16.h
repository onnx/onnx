/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for ScatterElements in default domain from version 15 to 16

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct ScatterElements_15_16 final: public Adapter {
  explicit ScatterElements_15_16() : Adapter("ScatterElements", OpSetID(15), OpSetID(16)) {}

  void adapt_scatterelements_15_16(std::shared_ptr<Graph>, Node* node) const {
    Symbol reduction = Symbol("reduction");
    node->s_(reduction, "none");
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_scatterelements_15_16(graph, node);
    return node;
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
