/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for Pad in default domain from version 10 to 11

#pragma once

namespace ONNX_NAMESPACE { namespace version_conversion {

class Pad_10_11 final : public Adapter {
  public:
    explicit Pad_10_11()
      : Adapter("Pad", OpSetID(10), OpSetID(11)) {}

    void adapt_pad_10_11(std::shared_ptr<Graph> graph, Node* node) const {
      // Turn pads attribute into input
      Tensor t_pads;
      t_pads.elem_type() = TensorProto_DataType_INT64;
      int input_rank = node->inputs()[0]->sizes().size();
      t_pads.sizes() = std::vector<int64_t> {2 * input_rank};
      auto& data_pads = t_pads.int64s();
      for (int64_t shape : node->is(kpads)) {
        data_pads.emplace_back(shape);
      }
      Value* v_pads = graph->addInitializerAndInput(t_pads);
      node->addInput(v_pads);
      node->removeAttribute(kpads);
      // Turn value attribute into input
      Tensor t_value;
      t_value.elem_type() = TensorProto_DataType_FLOAT;
      auto& data_value = t_value.floats();
      data_value.emplace_back(node->f(kvalue));
      Value* v_value = graph->addInitializerAndInput(t_value);
      node->addInput(v_value);
      node->removeAttribute(kvalue);
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
	    adapt_pad_10_11(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
