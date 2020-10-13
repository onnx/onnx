// Adapter for Slice in default domain from version 9 to 10

#pragma once

namespace ONNX_NAMESPACE { namespace version_conversion {

class Slice_9_10 final : public Adapter {
  public:
    explicit Slice_9_10()
      : Adapter("Slice", OpSetID(9), OpSetID(10)) {}

    void attrToInput(
        std::shared_ptr<Graph> graph,
        Node* node,
        const std::vector<int64_t>& attr) const {
      Tensor t;
      t.elem_type() = TensorProto_DataType_INT64;
      t.sizes() = std::vector<int64_t>{static_cast<int64_t>(attr.size())};
      auto& data = t.int64s();
      for (auto a : attr) {
        data.emplace_back(a);
      }
      Value* v;
      v = graph->addInitializerAndInput(t);
      node->addInput(v);
    }

    void adapt_slice_9_10(std::shared_ptr<Graph> graph, Node* node) const {
      attrToInput(graph, node, node->is(kstarts));
      node->removeAttribute(kstarts);
      attrToInput(graph, node, node->is(kends));
      node->removeAttribute(kends);

      if (node->hasAttribute(kaxes)){
        attrToInput(graph, node, node->is(kaxes));
        node->removeAttribute(kaxes);
      }
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_slice_9_10(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
