// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//	 A = Constant()
// After:
//	 A is in the initializer list
//
//	 this pass can handle the case satisfy all following conditions:
//	   condition 1: A is the output of a Constant node

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct ExtractConstantToInitializer final : public OptimizePass {
  explicit ExtractConstantToInitializer()
      : OptimizePass("extract_constant_to_initializer", API_TYPE::IR) {}

  void extract_constant_to_initializer(Graph& graph) {
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(
          n, [this](Graph& g) { extract_constant_to_initializer(g); });
      if (n->kind() == kConstant) {
        const auto name = n->output()->uniqueName();
        Tensor t = n->t(kvalue);

        // add a new graph input
        Value* input = graph.addInput();
        input->setUniqueName(name);
        input->setSizes({t.sizes().begin(), t.sizes().end()});
        input->setElemType(t.elem_type());
        n->output()->replaceAllUsesWith(input);

        // copy the tensor to initializer
        t.setName(name);
        graph.addInitializer(std::move(t), name);

        it.destroyCurrent();
      }
    }
  }

  void optimize(Graph& graph) override {
    extract_constant_to_initializer(graph);
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
