// Adapter for Upsample in default domain from version 8 to 9

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct Upsample_8_9 final: public Adapter {
  explicit Upsample_8_9() : Adapter("Upsample", OpSetID(8), OpSetID(9)) {}

  void adapt_upsample_8_9(std::shared_ptr<Graph> graph, Node* node) const {

    Symbol input_dirs = Symbol("scales");
    int dim = (int)(node->fs(kscales).size());
    Tensor t;
    t.elem_type() = TensorProto_DataType_FLOAT;
    t.sizes() = std::vector<int64_t>{dim};
    auto& data = t.floats();

    if (node->hasAttribute(input_dirs)) {
      for(double scale : node->fs(kscales)) {
        data.emplace_back((float)scale);
      }

      Value* v = graph->addInitializerAndInput(t, "scales");
      std::vector<Dimension> new_sizes {Dimension(dim)};
      v->setSizes(new_sizes);
      node->addInput(v);
      node->removeAttribute(kscales);
      }
    }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_upsample_8_9(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
