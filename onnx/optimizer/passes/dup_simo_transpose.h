// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct DupSIMOTranspose final : public PredicateBasedPass {
  explicit DupSIMOTranspose()
      : PredicateBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "dup_simo_transpose";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kTranspose && node->output()->uses().size() > 1;
  }
  bool runTransform(Node* node, Graph& g, NodeDestroyType& destroy_current)
      override {

    if(node->inputs().size() > 1)
      return false;

    std::cout << "transpose FOUND " << node->name() << " fanout "  << node->output()->uses().size() << std::endl;

    Node* transpose_predecessor = node->inputs()[0]->node();
    std::cout << "    SIMO node: " << node->name() << " pred: " << transpose_predecessor->name() << std::endl;
    
    for(int i = 1; i < node->output()->uses().size(); i++ ) {
      std::cout << "    SIMO1-user " << node->output()->uses()[i].user->name() << std::endl;

      Node* new_transpose = g.create(kTranspose, 1);
      std::cout << "    SIMO2 " << std::endl;

      // Copy the transpose permutation
      new_transpose->copyAttributes(*node);

      new_transpose->addInput(transpose_predecessor->output());
      std::cout << "    SIMO3 " << std::endl;
      std::cout << "    SIMO3a " << node->output()->uniqueName() << " In: " << node->input()->uniqueName() << std::endl;
      std::cout << "    SIMO3b " << node->output()->uses()[i].user->name() << std::endl;

      std::cout << "    SIMO4pre " << std::endl;
      node->output()->uses()[i].user->replaceInput(0, new_transpose->output());
      std::cout << "    SIMO4post " << std::endl;

      new_transpose->insertAfter(transpose_predecessor);
      std::cout << "    SIMO5 " << std::endl;
    }

    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
