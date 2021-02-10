/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Upsample in default domain from version 9 to 10

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct Upsample_9_10 final: public Adapter {
  explicit Upsample_9_10() : Adapter("Upsample", OpSetID(9), OpSetID(10)) {}

  void adapt_upsample_9_10(std::shared_ptr<Graph> graph, Node* node) const {

    const ArrayRef<Value*>& inputs = node->inputs();

    ONNX_ASSERTM(inputs.size() == 2, "Upsample in opset 9 needs to have 2 inputs.");
    std::string scale_input_name = node->inputs()[1]->uniqueName();

    Value* x = nullptr;
    Value* scales = nullptr;
    for(size_t i = 0; i < node->inputs().size(); i++) {
      if(node->inputs()[i]->uniqueName() == "X")
        x = node->inputs()[i];
      else if(node->inputs()[i]->uniqueName() == "Scales")
        scales = node->inputs()[i];
    }
    ONNX_ASSERTM(x && scales, "x or scale are not defined in Upsample input");

    Node* resizeNode = graph->create(kResize, 1);
    resizeNode->addInput(x);
    resizeNode->addInput(scales);
    resizeNode->insertAfter(node);

    ONNX_ASSERTM(node->outputs().size() == 1, "Upsample need to be only one output")
    node->output()->replaceAllUsesWith(resizeNode->output());

    node->destroy();
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_upsample_9_10(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
