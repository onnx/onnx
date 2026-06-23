// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for Attention in default domain from version 25 to 24

#pragma once

#include <cinttypes>
#include <cstdint>
#include <memory>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class Attention_25_24 final : public Adapter {
 public:
  explicit Attention_25_24() : Adapter("Attention", OpSetID(25), OpSetID(24)) {}

  void adapt_attention_25_24(const std::shared_ptr<Graph>& /*unused*/, Node* node) const {
    // local_window_size is new in opset 25; only -1 (disabled) is representable
    // in opset 24. Reject any other value.
    if (node->hasAttribute(klocal_window_size)) {
      int64_t val = node->i(klocal_window_size);
      if (val != -1) {
        ONNX_ASSERTM(
            false,
            "Attention 25->24 downgrade: local_window_size must be -1 (disabled) "
            "for conversion to opset 24, got %" PRId64
            ". "
            "Sliding window attention (local_window_size > 0) is not representable "
            "in opset 24.",
            val);
      }
      node->removeAttribute(klocal_window_size);
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_attention_25_24(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
