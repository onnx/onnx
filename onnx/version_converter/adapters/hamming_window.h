// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <unordered_map>

#include "onnx/common/ir_pb_converter.h"
#include "onnx/defs/schema.h"
#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE::version_conversion {

// Coefficients change at opset 29. Expand the source version's function rather
// than relabeling the node or applying a rounded affine correction to its output.
class HammingWindowAdapter final : public Adapter {
 public:
  HammingWindowAdapter(const OpSetID& initial, const OpSetID& target) : Adapter("HammingWindow", initial, target) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    const auto* schema = OpSchemaRegistry::Schema("HammingWindow", initial_version().version(), "");
    ONNX_ASSERT(schema != nullptr)
    const auto* function = schema->GetFunction();
    ONNX_ASSERT(function != nullptr)

    ModelProto model;
    model.set_ir_version(IR_VERSION);
    model.mutable_opset_import()->CopyFrom(function->opset_import());
    auto* body = model.mutable_graph();
    body->set_name("hamming_window_conversion");
    auto* input = body->add_input();
    input->set_name(function->input(0));
    auto* input_type = input->mutable_type()->mutable_tensor_type();
    input_type->set_elem_type(node->input()->elemType());
    input_type->mutable_shape();
    auto* output = body->add_output();
    output->set_name(function->output(0));
    output->mutable_type()->mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    body->mutable_node()->CopyFrom(function->node());
    for (auto& expanded_node : *body->mutable_node()) {
      for (auto& attribute : *expanded_node.mutable_attribute()) {
        if (attribute.has_ref_attr_name()) {
          const auto& name = attribute.ref_attr_name();
          const Symbol symbol{name};
          const auto value =
              node->hasAttribute(symbol) ? node->i(symbol) : schema->attributes().at(name).default_value.i();
          attribute.clear_ref_attr_name();
          attribute.set_i(value);
        }
      }
    }

    // The function contains only default-domain nodes with no subgraphs. All of
    // its primitive operator semantics are compatible across the 28/29 boundary.
    auto source = ImportModelProto(model);
    std::unordered_map<const Value*, Value*> values;
    values.emplace(source->inputs()[0], node->input());
    for (const auto* source_node : source->nodes()) {
      auto* clone = graph->create(source_node->kind(), source_node->outputs().size());
      clone->copyAttributes(*source_node);
      for (const auto* source_input : source_node->inputs()) {
        clone->addInput(values.at(source_input));
      }
      clone->insertBefore(node);
      for (size_t i = 0; i < source_node->outputs().size(); ++i) {
        values.emplace(source_node->outputs()[i], clone->outputs()[i]);
      }
    }
    auto* replacement = values.at(source->outputs()[0]);
    node->output()->replaceAllUsesWith(replacement);
    node->destroy();
    return replacement->node();
  }
};

} // namespace ONNX_NAMESPACE::version_conversion
