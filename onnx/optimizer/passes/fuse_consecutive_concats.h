// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConsecutiveConcats final : public PredicateBasedPass {
  explicit FuseConsecutiveConcats()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Partial,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_consecutive_concats";
  }

  void insertInput(Node* node, size_t i, Value* value) {
    const auto input_size = node->inputs().size();
    if (i == input_size) {
      node->addInput(value);
    } else {
      for (size_t j = input_size - 1; j >= i; j--) {
        Value* cur_input = node->input(j);
        if (j == input_size - 1) {
          node->addInput(cur_input);
        } else {
          node->replaceInput(j + 1, cur_input);
        }
      }
      node->replaceInput(i, value);
    }
  }

  bool patternMatchPredicate(Node* node) override {
    // we don't check if our concat node has inputs which are also concat nodes
    // because this requires a for loop through the inputs. If it turns out
    // there is then we still have to do a for loop in the runTransform portion
    // of the code. In order not to waste a loop we don't check the real pattern
    // match condition.
    return node->kind() == kConcat && node->hasAttribute(kaxis);
  }
  bool runTransform(Node* concat_node, Graph&, NodeDestroyType& destroy_current)
      override {
    destroy_current = NodeDestroyType::DestroyZero;
    bool transform_ran = false;
    for (size_t i = 0; i < concat_node->inputs().size(); i++) {
      Value* cur_input_value = concat_node->inputs()[i];
      Node* cur_input_node = cur_input_value->node();
      if (cur_input_node->kind() == kConcat &&
          cur_input_value->uses().size() == 1 &&
          cur_input_node->hasAttribute(kaxis) &&
          cur_input_node->i(kaxis) == concat_node->i(kaxis)) {
        transform_ran = true;
        // Inserts n inputs of cur_input_node at index i+1~i+1+(n-1), 
        // and remove cur_input_node at index i. 
        // As a result, cur_input_node is replaced by its inputs inplace, 
        // instead of always appending its inputs at the end.
        for (size_t j = 0; j < cur_input_node->inputs().size(); j++) {
          Value* value = cur_input_node->input(j);
          insertInput(concat_node, i + 1 + j, value);
        }
        concat_node->removeInput(i);
        cur_input_node->destroy();
      }
    }
    return transform_ran;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
