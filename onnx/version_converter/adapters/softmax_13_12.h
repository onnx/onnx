// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Softmax, LogSoftmax and Hardmax in default domain from version 13 to 12

#pragma once

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
    // Opset 13 added bfloat16 to these operators.
    ONNX_ASSERTM(
        node->inputs()[0]->elemType() != TensorProto_DataType_BFLOAT16,
        "BFLOAT16 input of operator '",
        name(),
        "' is unallowed for Opset Version 12.")

    const int input_rank = static_cast<int>(node->inputs()[0]->sizes().size());
    const int64_t axis = node->hasAttribute(kaxis) ? node->i(kaxis) : -1;
    if (axis == -1) {
      // Last axis (the opset 13 default). With an unknown rank this stays -1, which
      // opset 12 also reads as the last axis.
      node->i_(kaxis, input_rank > 0 ? input_rank - 1 : -1);
      return;
    }
    ONNX_ASSERTM(input_rank > 0, name(), " with a non-last axis needs a known input rank")
    ONNX_ASSERTM(
        axis >= -input_rank && axis < input_rank, name(), " axis ", axis, " is out of range for rank ", input_rank)
    const int new_axis = static_cast<int>(axis < 0 ? axis + input_rank : axis);
    node->i_(kaxis, new_axis);
    if (new_axis == input_rank - 1) {
      return;
    }

    // Opset 12 coerces the input to 2D at axis, so it matches opset 13 only on the
    // last axis. Otherwise swap axis with the last one around the op.
    std::vector<int64_t> perm(input_rank);
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[new_axis], perm[input_rank - 1]);

    Node* transpose_in = graph->create(Symbol("Transpose"));
    transpose_in->addInput(node->inputs()[0]);
    transpose_in->is_(kperm, std::vector<int64_t>(perm));
    transpose_in->insertBefore(node);
    node->replaceInput(0, transpose_in->output());
    node->i_(kaxis, input_rank - 1);

    Node* transpose_out = graph->create(Symbol("Transpose"));
    transpose_out->is_(kperm, std::move(perm));
    transpose_out->output()->setElemType(node->output()->elemType());
    if (node->output()->has_sizes()) {
      transpose_out->output()->setSizes(node->output()->sizes());
    }
    transpose_out->insertAfter(node);
    ReplaceValueKeepingName(*graph, node->output(), transpose_out->output());
    transpose_out->addInput(node->output());
    node->output()->wipeSizes();
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_softmax_13_12(graph, node);
    return node;
  }
};

} // namespace ONNX_NAMESPACE::version_conversion
