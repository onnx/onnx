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
      // Get shape from initializer or constant operator, not actual shape
      // Identify whether we have a Constant Op or an Initializer
      Node* node_ptr = inputs[1]->node();
      if (node_ptr->kind() == kConstant) {
        // Get value attribute of kConstant
        const std::vector<int64_t>& int64s = node_ptr->t(kvalue).int64s();
        if (int64s.empty()) {
          // Also handle raw data
          int64_t* raw = (int64_t*) node_ptr->t(kvalue).raw().c_str();
          node->is_(kshape, std::vector<int64_t>(raw, raw + (sizeof(raw)/sizeof(
                    raw[0]))));
        } else {
          node->is_(kshape, std::forward<const std::vector<int64_t>>(int64s));
        }
        // If Constant node isn't used anywhere else, remove it
        Value* const_val = inputs[1];
        node->removeInput(1);
        if (const_val->uses().size() == 1) {
          node_ptr->destroy();
        }
      } else {
        // Get Value name, find Initializer with same name
        for (const auto& initializer: graph->initializers()) {
          if (initializer.name() == inputs[1]->uniqueName()) {
            node->is_(kshape, std::forward<const std::vector<int64_t>>(
                  initializer.int64s()));
            node->removeInput(1);
            // Remove initializer
            // TODO: Iterate through all inputs to detect whether others are the same
            bool otherUses = false;
            for (Node* node : graph->nodes()) {
              for (Value* input : node->inputs()) {
                if (input->uniqueName() == initializer.name()) {
                  otherUses = true;
                  break;
                }
              }
              if (otherUses) break;
            }
            if (!otherUses) graph->eraseInitializer(initializer.name());
          }
        }
      }
      ONNX_ASSERTM(node->hasAttribute(kshape),
          "No initializer or constant input to Reshape node found");
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
      adapt_reshape_5_4(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
