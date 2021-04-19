/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Adapter for TopK in default domain from version 9 to 10

#pragma once

namespace ONNX_NAMESPACE { namespace version_conversion {

class TopK_9_10 final : public Adapter {
  public:
    explicit TopK_9_10()
      : Adapter("TopK", OpSetID(9), OpSetID(10)) {}

    void adapt_topk_9_10(std::shared_ptr<Graph> graph, Node* node) const {
      Tensor t;
      t.elem_type() = TensorProto_DataType_INT64;
      t.sizes() = std::vector<int64_t> {1};
      auto& data = t.int64s();
      data.emplace_back(node->i(kk));
      
      Value* v = graph->addInitializerAndInput(t);
      v->setSizes(std::vector<Dimension> {Dimension(1)});
      node->addInput(v);

      node->removeAttribute(kk);
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_topk_9_10(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
