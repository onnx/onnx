// Adapter for Add in default domain from version 7 to 6

#pragma once

#include "onnx/version_converter/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct Add_7_6 final : public Adapter {
  explicit Add_7_6()
    : Adapter("Add", make_opsetid("", 7), make_opsetid("", 6)) {
    }

  void adapt_add_7_6(Graph& graph, Node& node) {
    // Verify that broadcasts are allowed in limited spec of opset version 6
    // Multidirectional broadcasting, as defined in Broadcasting.md
    // MathDocGenerator provides differences
    // Main change: encode broadcasting commands as explicit attribute
    ArrayRef<Value*> inputs = node.inputs();
    std::vector<Dimension> A_sizes = inputs[0]->sizes();
    std::vector<Dimension> B_sizes = inputs[1]->sizes();
    // Determine if inputs are of different sizes
    if (inputs.size() == 2 && !A_sizes.equals(B_sizes)) {
      // Assert that first input is larger than or equal to the second
      ONNX_ASSERTM(A_sizes.size() >= B_sizes.size(), "Opset Version
          6 does not support numpy-style broadcasting: input A must be larger
          than B");
      // Determine what the broadcast dimension is - if necessary
      // Unnecessary if 1) all inputs are 1, 2) B is empty, 3) dimensions match
      // in reverse (assume this is required for model to compile in the first place?)
      int axis = A_sizes.size() - B_sizes.size();
      for (int i = B_sizes.size() - 1; i >= 0; i--) {
        if (axis >= 0) {
          if (B_sizes[i] == A_sizes[axis + i] || B_sizes[i] == 1) {
            continue;
          } else {
            // Try decreasing the axis
            axis--;
            i++;
          }
        }
      }
      // Assert that final state is well-formed
      if (axis != A_sizes.size() - B_sizes.size()) {
        // Need to add axis attribute
      }
      // Add broadcast attribute

    }
  }

  void adapt(Graph& graph, Node& node) override {
    adapt_add_7_6(graph, node);
  }

  OpSetID make_opsetid(std::string domain, int version) {
    OpSetID ret;
    ret.domain = domain;
    ret.version = version;
    return ret;
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
