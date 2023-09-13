// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class AxisAttributeToInput : public Adapter {
 public:
  explicit AxisAttributeToInput(const std::string& op_name, const OpSetID& initial, const OpSetID& target)
      : Adapter(op_name, initial, target) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    if (node->hasAttribute(kaxis)) {
      AttrToInput(graph, node, node->i(kaxis));
      node->removeAttribute(kaxis);
    }
    return node;
  }

private:
  void AttrToInput(std::shared_ptr<Graph> graph, Node* node, int64_t axis) const {
    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.sizes() = std::vector<int64_t>{};
    auto& data = t.int64s();
    data.emplace_back(axis);

    Node* constant = graph->create(kConstant);
    constant->insertBefore(node);
    constant->t_(kvalue, t);
    node->addInput(constant->output());
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
