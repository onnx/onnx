// Adapter for Reshape in default domain from version 5 to 4

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Reshape_5_4 final : public Adapter {
  public:
    explicit Reshape_5_4()
      : Adapter("Reshape", OpSetID(5), OpSetID(4)) {}

    void adapt_reshape_5_4(std::shared_ptr<Graph> graph, Node* node) const {
      // Identify if shape is statically determined; if so, feed as attribute
      const ArrayRef<Value*>& inputs = node->inputs();
      // Confirm that second input will always be shape (and first will be data)
      const std::vector<Dimension>& shapeRef = inputs[1]->sizes();
      ONNX_ASSERTM(!shapeRef.empty(), "Output shape must be provided "
        "as a static input.");
      std::vector<int64_t> shape;
      for (int i = 0; i < (int) shapeRef.size(); i++) {
        shape.emplace_back(shapeRef[i].dim);
      }
      node->removeInput(1);
      node->is_(kshape, std::forward<const std::vector<int64_t>>(shape));
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_reshape_5_4(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
