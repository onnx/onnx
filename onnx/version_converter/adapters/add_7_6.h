// Adapter for Add in default domain from version 7 to 6

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct Add_7_6 final : public Adapter {
  explicit Add_7_6()
    : Adapter("Add", OpSetID(7), OpSetID(6)) {
    }

  void adapt_add_7_6(std::shared_ptr<Graph> graph, Node* node) const {
    // Verify that broadcasts are allowed in limited spec of opset version 6
    // Multidirectional broadcasting, as defined in Broadcasting.md
    // MathDocGenerator provides differences
    // Main change: encode broadcasting commands as explicit attribute
    ArrayRef<Value*> inputs = node->inputs();
    ONNX_ASSERTM(inputs.size() == 2, "Add in opset version 6 can only broadcast"
      " between 2 inputs");
    std::vector<Dimension> A_sizes = inputs[0]->sizes();
    std::vector<Dimension> B_sizes = inputs[1]->sizes();
    // Determine if inputs are of different sizes
    // TODO: Dimension doesn't have equality implemented; may need to do this manually
    bool equalDims = false;
    if (A_sizes.size() == B_sizes.size()) {
      equalDims = true;
      for (int i = 0; i < A_sizes.size(); i++) {
        if (A_sizes[i].dim != B_sizes[i].dim) {
          equalDims = false;
        }
      }
    }
    if (!equalDims) {
      // Ensure that first input is larger than or equal to the second
      if(A_sizes.size() < B_sizes.size()) {
        // Handle switching input order
        Value* A = node->replaceInput(0, inputs[1]);
        node->replaceInput(1, A);
        A_sizes = B_sizes;
        B_sizes = inputs[1]->sizes();
      }
      // Determine what the broadcast dimension is - if necessary
      // Unnecessary if 1) all inputs are 1, 2) B is empty, 3) dimensions match
      // in reverse (assume this is required for model to compile in the first place?)
      int axis = A_sizes.size() - B_sizes.size();
      for (int i = B_sizes.size() - 1; i >= 0; i--) {
        ONNX_ASSERTM(axis >= 0, "Inputs are not broadcastable: no positive axis "
          "found to align dimensions.");
        if (B_sizes[i].dim == A_sizes[axis + i].dim || B_sizes[i].dim == 1) {
          continue;
        } else {
          // Try decreasing the axis
          axis--;
          i++;
        }
      }
      if (axis != A_sizes.size() - B_sizes.size()) {
        // Add axis attribute
        node->i_(kaxis, axis);
      }
      // If conditional is not fulfilled, we have a default broadcast
      // Add broadcast attribute
      node->i_(kbroadcast, 1);
    }
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_add_7_6(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
