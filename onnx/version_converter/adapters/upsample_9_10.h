// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Upsample in default domain from version 9 to 10

#pragma once

#include <memory>
#include <string>

#include "onnx/common/ir.h"
#include "onnx/version_converter/adapters/adapter.h"
namespace ONNX_NAMESPACE {
namespace version_conversion {

class Upsample_9_10 final : public Adapter {
 public:
  explicit Upsample_9_10() : Adapter("Upsample", OpSetID(9), OpSetID(10)) {}

  Node* adapt_upsample_9_10(const std::shared_ptr<Graph>& graph, Node* node) const {
    std::string mode = node->hasAttribute(BuiltinSymbol::kmode) ? node->s(BuiltinSymbol::kmode) : "nearest";

    // Replace the node with an equivalent Resize node
    Node* resize = graph->create(BuiltinSymbol::kResize);
    resize->s_(BuiltinSymbol::kmode, mode);
    resize->addInput(node->inputs()[0]);
    resize->addInput(node->inputs()[1]);
    node->replaceAllUsesWith(resize);

    resize->insertBefore(node);
    node->destroy();

    return resize;
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    return adapt_upsample_9_10(graph, node);
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
