/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Softmax amd LogSoftmax in default domain from version 12 to 13

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Softmax_12_13 final : public Adapter {
 public:
  explicit Softmax_12_13(const std::string& op_name) : Adapter(op_name, OpSetID(12), OpSetID(13)) {}

  void adapt_softmax_12_13(std::shared_ptr<Graph> graph, Node* node) const {
    int old_axis = node->hasAttribute(kaxis) ? node->i(kaxis) : 1;
    int input_rank = node->inputs()[0]->sizes().size();

    if (old_axis < 0)
      old_axis = input_rank + old_axis;

    if (old_axis == input_rank - 1)
      node->i_(kaxis, -1);
    else {
      // Insert Flatten node before softmax
      Node* flatten = graph->create(kFlatten);
      flatten->addInput(node->inputs()[0]);
      flatten->insertBefore(node);
      flatten->i_(kaxis, old_axis);
      node->replaceInput(0, flatten->output());

      if (old_axis == 0) {
        node->i_(kaxis, 1);
      } else {
        node->i_(kaxis, -1);
      }

      // Insert Reshape node after softmax
      const std::string original_output_name = node->output()->uniqueName();
      const use_list original_uses(node->output()->uses());
      node->output()->setUniqueName(original_output_name + "_intermediate");
      Node* reshape = graph->create(kReshape);
      reshape->addInput(node->outputs()[0]);
      reshape->output()->setUniqueName(original_output_name);
      reshape->insertAfter(node);

      // Set shape input of Reshape
      const std::vector<Dimension>& target_shape = flatten->inputs()[0]->sizes();

      ONNX_ASSERTM(
          target_shape.size() != 0,
          "Version conversion for Softmax failed because "
          "input shape is unknown.");

      Tensor t;
      t.elem_type() = TensorProto_DataType_INT64;
      t.sizes() = std::vector<int64_t>{static_cast<int64_t>(target_shape.size())};
      auto& data = t.int64s();
      for (Dimension dim : target_shape) {
        data.emplace_back(dim.dim);
      }
      Node* constant = graph->create(kConstant);
      constant->insertBefore(node);
      constant->t_(kvalue, t);
      reshape->addInput(constant->output());

      // Fix outputs & wiring
      node->output()->wipeSizes();
      reshape->output()->setSizes(target_shape);
      reshape->output()->setElemType(node->output()->elemType());
      for (Use u : original_uses) {
        u.user->replaceInputWith(node->output(), reshape->output());
      }
      for (size_t i = 0; i < graph->outputs().size(); i++) {
        if (graph->outputs()[i]->uniqueName() == node->output()->uniqueName()) {
          graph->return_node()->replaceInput(i, reshape->output());
        }
      }
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_softmax_12_13(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
