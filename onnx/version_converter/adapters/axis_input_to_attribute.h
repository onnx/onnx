// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class AxisInputToAttribute : public Adapter {
 public:
  explicit AxisInputToAttribute(const std::string& op_name, const OpSetID& initial, const OpSetID& target)
      : Adapter(op_name, initial, target) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    // Identify if axis is statically determined; if so, feed as attribute
    const ArrayRef<Value*>& inputs = node->inputs();
    // Get axis from initializer or constant operator
    // Identify whether we have a Constant Op or an Initializer
    Value* const_val = inputs[1];
    Node* node_ptr = const_val->node();
    if (node_ptr->kind() == kConstant) {
      // Get value attribute of kConstant
      const std::vector<int64_t>& int64s = node_ptr->t(kvalue).int64s();
      if (int64s.empty()) {
        // Also handle raw data
        std::string raw_data = node_ptr->t(kvalue).raw();
        ONNX_ASSERTM(
            raw_data.size() != 0 && raw_data.size() % 8 == 0,
            "Raw Data must be non-empty and size must be a multiple of 8");
        int64_t* raw = (int64_t*)const_cast<char*>(raw_data.c_str());
        node->i_(kaxis, static_cast<int64_t>(raw);
      } else {
        node->i_(kaxis, static_cast<int64_t>(int64s.at(0)));
      }
      // If Constant node isn't used anywhere else, remove it
      node->removeInput(1);
      if (const_val->uses().size() < 1) {
        node_ptr->destroy();
      }
    } else {
      // Get Value name, find Initializer with same name
      for (const auto& initializer : graph->initializers()) {
        if (initializer.name() == inputs[1]->uniqueName()) {
          node->i_(kaxis, static_cast<int64_t>(initializer.int64s().at(0))));
          node->removeInput(1);
          // Remove initializer
          if (const_val->uses().size() < 1)
            graph->eraseInitializerAndInput(const_val);
          break;
        }
      }
    }
    ONNX_ASSERTM(node->hasAttribute(kaxis), "No initializer or constant input to node found");
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
