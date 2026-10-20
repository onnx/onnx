// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Softmax and LogSoftmax in default domain from version 13 to 12

#pragma once

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {

class Softmax_13_12 final : public Adapter {
 public:
  explicit Softmax_13_12(const std::string& op_name) : Adapter(op_name, OpSetID(13), OpSetID(12)) {}

  void adapt_softmax_13_12(const std::shared_ptr<Graph>& graph, Node* node) const {
    ONNX_ASSERTM(node->inputs().size() >= 1, "Softmax node must have at least 1 input")

    int64_t new_axis = node->hasAttribute(kaxis) ? node->i(kaxis) : -1;
    int input_rank = static_cast<int>(node->inputs()[0]->sizes().size());

    // Handle Softmax on the last axis (default behavior in opset 13)
    if (new_axis == -1) {
      new_axis = input_rank - 1;
      node->i_(kaxis, new_axis);
    } else if (new_axis < 0) {
      // Handle negative axis by converting it to positive
      new_axis = input_rank + new_axis;
      node->i_(kaxis, new_axis);
    }

    // Opset 12 coerces the input to 2D at axis, so it matches opset 13 only on the
    // last axis. Otherwise swap axis with the last one around the op.
    // Flatten produces a rank-two tensor, but axis 0 still differs between
    // opsets. Its presence does not justify bypassing the transpose pair.
    if (new_axis != input_rank - 1) {
      ONNX_ASSERTM(input_rank > 0, "Softmax with a non-last axis needs a known input rank")
      ONNX_ASSERTM(new_axis >= 0 && new_axis < input_rank, "Softmax axis is out of range for the input rank")
      std::vector<int64_t> perm(input_rank);
      std::iota(perm.begin(), perm.end(), 0);
      std::swap(perm[new_axis], perm[input_rank - 1]);

      Node* transpose_in = graph->create(Symbol("Transpose"));
      transpose_in->addInput(node->inputs()[0]);
      transpose_in->is_(kperm, std::vector<int64_t>(perm));
      transpose_in->insertBefore(node);
      node->replaceInput(0, transpose_in->output());
      node->i_(kaxis, input_rank - 1);

      const use_list original_uses(node->output()->uses());
      const std::string original_output_name = node->output()->uniqueName();
      node->output()->setUniqueName(original_output_name + "_transposed");
      Node* transpose_out = graph->create(Symbol("Transpose"));
      transpose_out->addInput(node->output());
      transpose_out->is_(kperm, std::move(perm));
      transpose_out->output()->setUniqueName(original_output_name);
      transpose_out->output()->setElemType(node->output()->elemType());
      transpose_out->output()->setSizes(node->output()->sizes());
      node->output()->wipeSizes();
      transpose_out->insertAfter(node);
      for (Use u : original_uses) {
        u.user->replaceInputWith(node->output(), transpose_out->output());
      }
      for (size_t i = 0; i < graph->outputs().size(); i++) {
        if (graph->outputs()[i] == node->output()) {
          graph->return_node()->replaceInput(i, transpose_out->output());
        }
      }
      return;
    }

    // Neighboring Flatten/Reshape nodes are part of the model's semantics.
    // Preserve them, including those inserted by the 12 -> 13 adapter: their
    // last-axis, rank-two Softmax is already equivalent in opset 12.
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_softmax_13_12(graph, node);
    return node;
  }
};

} // namespace ONNX_NAMESPACE::version_conversion
