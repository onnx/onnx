/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Resize in default domain from version 10 to 11

#pragma once

namespace ONNX_NAMESPACE { namespace version_conversion {

class Resize_10_11 final : public Adapter {
  public:
    explicit Resize_10_11()
      : Adapter("Resize", OpSetID(10), OpSetID(11)) {}

    void adapt_resize_10_11(Node* node) const {
      Value* scales_input = node->inputs()[1];
      node->addInput(scales_input);
      Value* dummy = new Value(node, 1);
      dummy->setUniqueName("");
      node->replaceInput(1, dummy);
    }

    void adapt(std::shared_ptr<Graph> , Node* node) const override {
      adapt_resize_10_11(node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
