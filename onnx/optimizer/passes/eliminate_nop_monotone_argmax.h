// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.
#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

const std::unordered_set<NodeKind> monotone_node_no_axis_kind{kLog,
                                                              kExp,
                                                              kSqrt};

const std::unordered_set<NodeKind> monotone_node_axis_kind{kSoftmax,
                                                           kLogSoftmax};

struct EliminateNopMonotoneArgmax final : public PredicateBasedPass {
  explicit EliminateNopMonotoneArgmax()
      : PredicateBasedPass(
            PassType::Nop,
            PassEfficiency::Partial,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_monotone_argmax";
  }

  static inline bool satisfies_monotone_condition(int64_t axis, Node* node) {
    if (monotone_node_no_axis_kind.find(node->kind()) !=
        monotone_node_no_axis_kind.end()) {
      return true;
    }
    if (monotone_node_axis_kind.find(node->kind()) !=
        monotone_node_axis_kind.end()) {
      if (node->hasAttribute(kaxis)) {
        return axis == node->i(kaxis);
      }
    }
    return false;
  }

  bool patternMatchPredicate(Node* node) override {
    if (node->kind() == kArgMax) {
      if (node->hasAttribute(kaxis)) {
        auto node_axis = node->i(kaxis);
        return node->inputs().size() == 1 &&
            satisfies_monotone_condition(node_axis, node->input()->node());
      }
    }
    return false;
  }

  bool runTransform(Node* node, Graph&, NodeDestroyType&)
      override {
    Node* monotone_node = node->input()->node();
    if (monotone_node->output()->uses().size() == 1) {
      monotone_node->output()->replaceAllUsesWith(monotone_node->input());
      monotone_node->destroy();
      return true;
    }
    return false;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE