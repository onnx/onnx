// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Cast in default domain from version 9 to 8

#pragma once

#include <memory>
#include <vector>

#include "onnx/version_converter/adapters/type_restriction.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

static const std::vector<TensorProto_DataType> q_dq_20_unallowed_types = {
    TensorProto_DataType_UINT16,
    TensorProto_DataType_INT16};

class QuantizeLinear_21_20 final : public TypeRestriction {
 public:
  explicit QuantizeLinear_21_20()
      : TypeRestriction("QuantizeLinear", OpSetID(21), OpSetID(20), q_dq_20_unallowed_types) {}

  void adapt_quantize_linear_21_20(std::shared_ptr<Graph>, Node* node) const {
    if (node->inputs()[1]->sizes().size() > 1) {
      ONNX_ASSERTM(false, "Scale must be a vector or a number for Opset Version %d.", target_version().version())
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_type_restriction(graph, node);
    adapt_quantize_linear_21_20(graph, node);
    return node;
  }
};

class DequantizeLinear_21_20 final : public TypeRestriction {
 public:
  explicit DequantizeLinear_21_20()
      : TypeRestriction("DequantizeLinear", OpSetID(21), OpSetID(20), q_dq_20_unallowed_types) {}

  void adapt_dequantize_linear_21_20(std::shared_ptr<Graph>, Node* node) const {
    if (node->inputs()[1]->sizes().size() > 1) {
      ONNX_ASSERTM(false, "Scale must be a vector or a number for Opset Version %d.", target_version().version())
    }
  }

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_type_restriction(graph, node);
    adapt_dequantize_linear_21_20(graph, node);
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
