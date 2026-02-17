// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Adapter for ScatterElements and ScatterND in default domain from version 16 to 15.

#pragma once

#include <algorithm>
#include <cinttypes>
#include <string>
#include <vector>

#include "onnx/common/interned_strings.h"
#include "onnx/version_converter/adapters/adapter.h"
#include "onnx/version_converter/adapters/transformers.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class ScatterElements_16_15 : public Adapter {
 public:
  explicit ScatterElements_16_15(
      const OpSetID& initial,
      const OpSetID& target,
      const std::vector<TensorProto_DataType>& unallowed_types)
      : Adapter("ScatterElements", initial, target), unallowed_types_(unallowed_types) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_type_restriction(graph, node);
    RemoveAttribute(Symbol("reduction"), std::string("none"))(graph, node);
    return node;
  }

 private:
  std::vector<TensorProto_DataType> unallowed_types_;

  void adapt_type_restriction(const std::shared_ptr<Graph>& /*unused*/, const Node* node) const {
    for (const Value* input : node->inputs()) {
      isUnallowed(input);
    }
    for (const Value* output : node->outputs()) {
      isUnallowed(output);
    }
  }

  void isUnallowed(const Value* val) const {
    ONNX_ASSERTM(
        std::find(std::begin(unallowed_types_), std::end(unallowed_types_), val->elemType()) ==
            std::end(unallowed_types_),
        "DataType (%d) of Input or Output"
        " of operator '%s' is unallowed for Opset Version %" PRId64 ".",
        val->elemType(),
        name().c_str(),
        static_cast<int64_t>(target_version().version()))
  }
};

class ScatterND_16_15 : public Adapter {
 public:
  explicit ScatterND_16_15(
      const OpSetID& initial,
      const OpSetID& target,
      const std::vector<TensorProto_DataType>& unallowed_types)
      : Adapter("ScatterND", initial, target), unallowed_types_(unallowed_types) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_type_restriction(graph, node);
    RemoveAttribute(Symbol("reduction"), std::string("none"))(graph, node);
    return node;
  }

 private:
  std::vector<TensorProto_DataType> unallowed_types_;

  void adapt_type_restriction(const std::shared_ptr<Graph>& /*unused*/, const Node* node) const {
    for (const Value* input : node->inputs()) {
      isUnallowed(input);
    }
    for (const Value* output : node->outputs()) {
      isUnallowed(output);
    }
  }

  void isUnallowed(const Value* val) const {
    ONNX_ASSERTM(
        std::find(std::begin(unallowed_types_), std::end(unallowed_types_), val->elemType()) ==
            std::end(unallowed_types_),
        "DataType (%d) of Input or Output"
        " of operator '%s' is unallowed for Opset Version %" PRId64 ".",
        val->elemType(),
        name().c_str(),
        static_cast<int64_t>(target_version().version()))
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
