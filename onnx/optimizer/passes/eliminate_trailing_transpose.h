// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateTrailingTranspose final : public PredicateBasedPass {
  explicit EliminateTrailingTranspose()
      : PredicateBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {
    // this was not defined in the standard onnx strings
    mp_sym_ = Symbol("MaxPool");
    ave_sym_ = Symbol("AveragePool");
    crop_sym_ = Symbol("Crop");
  }

  std::string getPassName() const override {
    return "eliminate_trailing_transpose";
  }

  bool patternMatchPredicate(Node* node) override {
    if(node->kind() != kTranspose)
      return false;
    auto prev_node = node->inputs()[0]->node();
    bool res = (prev_node->kind() == kConv
		|| prev_node->kind() == kConvTranspose
		|| prev_node->kind() == ave_sym_
		|| prev_node->kind() == crop_sym_
		|| prev_node->kind() == mp_sym_);
    std::cout << "FOUND " << res << " " << node->name() << " " << prev_node->name() << " " << prev_node->kind().toString() << std::endl;
    return res;
  }

  bool runTransform(Node* node, Graph& graph, NodeDestroyType& destroy_current)
      override {
    destroy_current = NodeDestroyType::DestroyZero;
    auto base_node = node->inputs()[0];

    // check if base_node is only used by target
    if (base_node->uses().size() > 1) {
      std::cout << "  FAIL " << base_node->uses().size() << " " << node->name() << std::endl;
      return false;
    }

    if(base_node->node()->hasAttribute(Symbol("__perm"))) {
      return false;
    }

    std::vector<int64_t> perm = node->is(kperm);
    base_node->node()->is_(Symbol("__perm"), std::move(perm));

    // Don't assume that theres only one output.
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      node->outputs()[i]->replaceAllUsesWith(node->input(0));
    }

    destroy_current = NodeDestroyType::DestroyOne;    
    return true;
  }
 private:
    Symbol mp_sym_;
    Symbol ave_sym_;
    Symbol crop_sym_;
};


struct EliminateLeadingTranspose final : public PredicateBasedPass {
  explicit EliminateLeadingTranspose()
      : PredicateBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {
    mp_sym_ = Symbol("MaxPool");
    ave_sym_ = Symbol("AveragePool");
    crop_sym_ = Symbol("Crop");    
  }

  std::string getPassName() const override {
    return "eliminate_leading_transpose";
  }

  bool patternMatchPredicate(Node* node) override {
    if(node->kind() != kTranspose)
      return false;
    auto next_node = node->output()->uses()[0].user;
    bool res = (next_node->kind() == kConv
		|| next_node->kind() == kConvTranspose
		|| next_node->kind() == ave_sym_
		|| next_node->kind() == crop_sym_
		|| next_node->kind() == mp_sym_);
    std::cout << "FOUND-lead " << res << " " << node->name() << " next: " << next_node->kind().toString() << std::endl;
    return res;
  }

  bool runTransform(Node* node, Graph& graph, NodeDestroyType& destroy_current)
      override {
    destroy_current = NodeDestroyType::DestroyZero;

    // check if transpose node is only used by target
    if (node->outputs().size() != 1 || node->output()->uses().size() != 1) {
      std::cout << "  FAIL-lead " << node->outputs().size() << " " << node->output()->uses().size() << " " << node->name() << std::endl;
      return false;
    }
    auto base_node = node->output()->uses()[0].user;

    if (node->inputs().size() > 1) {
      std::cout << "  FAIL2-lead " << node->inputs().size() << " " << node->name() << std::endl;
      return false;
    }

    if(!base_node->hasAttribute(Symbol("__perm"))) {
      std::cout << "  FAIL3-lead " << std::endl;
      return false;
    }

    // Check that the permutations are complementary
    std::vector<int64_t> perm = node->is(kperm);
    std::vector<int64_t> base_perm = base_node->is(Symbol("__perm"));
    if(perm.size() != base_perm.size()) {
      std::cout << "  FAIL4-lead " << std::endl;
      return false;
    }      
      
    for(int i = 0; i < perm.size(); i++ ) {
      if(base_perm[perm[i]] != i) {
	std::cout << "  FAIL5-lead " << std::endl;
	return false;
      }
    }
    base_node->removeAttribute(Symbol("__perm"));
    base_node->i_(Symbol("__channels_last"), 1);

    // we verified that there is only one input and one output
    node->output()->replaceAllUsesWith(node->input(0));

    destroy_current = NodeDestroyType::DestroyOne;    
    return true;
  }
 private:
    Symbol mp_sym_;
    Symbol ave_sym_;
    Symbol crop_sym_;
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
