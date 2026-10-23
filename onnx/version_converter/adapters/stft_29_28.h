// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <utility>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {

class STFT_29_28 final : public Adapter {
 public:
  explicit STFT_29_28() : Adapter("STFT", OpSetID(29), OpSetID(28)) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    Value* signal = node->inputs()[0];
    ONNX_ASSERTM(signal->has_sizes(), "STFT conversion requires a known signal rank.")
    ONNX_ASSERTM(
        signal->sizes().size() == 2 || signal->sizes().size() == 3,
        "STFT conversion requires a rank-2 or rank-3 signal.")
    if (signal->sizes().size() == 3) {
      return node;
    }

    Tensor axes;
    axes.elem_type() = TensorProto_DataType_INT64;
    axes.sizes() = {1};
    axes.int64s() = {2};
    Node* constant = graph->create(kConstant);
    constant->t_(kvalue, axes);
    constant->insertBefore(node);

    Node* unsqueeze = graph->create(kUnsqueeze);
    unsqueeze->addInput(signal);
    unsqueeze->addInput(constant->output());
    unsqueeze->insertBefore(node);
    auto sizes = signal->sizes();
    sizes.emplace_back(1);
    unsqueeze->output()->setSizes(std::move(sizes));
    unsqueeze->output()->setElemType(signal->elemType());
    node->replaceInput(0, unsqueeze->output());
    return node;
  }
};

} // namespace ONNX_NAMESPACE::version_conversion
