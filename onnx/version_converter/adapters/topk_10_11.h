/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for TopK in default domain from version 10 to 11

#pragma once

namespace ONNX_NAMESPACE { namespace version_conversion {

class TopK_10_11 final : public Adapter {
  public:
    explicit TopK_10_11()
      : Adapter("TopK", OpSetID(10), OpSetID(11)) {}

    void adapt_topk_10_11(std::shared_ptr<Graph> graph, Node* node) const {
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_topk_10_11(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
