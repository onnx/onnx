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
#include "onnx/common/assertions.h"
#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct ExtractConstantToInitializer final : public PredicateBasedPass {
  explicit ExtractConstantToInitializer()
      : PredicateBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::Memory) {}

  std::string getPassName() const override {
    return "extract_constant_to_initializer";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kConstant;
  }

  bool runTransform(Node* node, Graph& graph, NodeDestroyType& destroy_current)
      override {
    const auto name = node->output()->uniqueName();
    Tensor t = node->t(kvalue);
    Value* new_init = graph.addInitializerAndInput(t, name);
    node->output()->replaceAllUsesWith(new_init);
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
