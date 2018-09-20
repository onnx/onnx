// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

const std::unordered_set<NodeKind> monotone_node_no_axis_kind{kLog,
                                                              kExp,
                                                              kSqrt};
const std::unordered_set<NodeKind> monotone_node_axis_kind{kSoftmax,
                                                           kLogSoftmax};

struct EliminateNopMonotoneArgmax final : public OptimizePass {
  explicit EliminateNopMonotoneArgmax()
      : OptimizePass("eliminate_nop_monotone_argmax", API_TYPE::IR) {}

  static inline bool satisfies_monotone_condition(int64_t axis, Node* node) {
    if (monotone_node_no_axis_kind.find(node->kind()) !=
        monotone_node_no_axis_kind.end()) {
      return true;
    }
    if (monotone_node_axis_kind.find(node->kind()) !=
        monotone_node_axis_kind.end()) {
      return axis == node->i(kaxis);
    }
    return false;
  }

  bool eliminate_nop_monotone_argmax(Graph& graph) {
    bool mark_change = false;
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      mark_change |= DescendOnGraphAttributesMarkChange(
          n, [this](Graph& g) { return eliminate_nop_monotone_argmax(g); });

      if (n->kind() == kArgMax && n->inputs().size() == 1 &&
          satisfies_monotone_condition(n->i(kaxis), n->input()->node())) {
        Node* monotone_node = n->input()->node();
        if (monotone_node->output()->uses().size() == 1) {
          monotone_node->output()->replaceAllUsesWith(monotone_node->input());
          monotone_node->destroy();
          mark_change = true;
        }
      }
    }
    return mark_change;
  }

  void optimize(Graph& graph) override {
    while (eliminate_nop_monotone_argmax(graph))
      ;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
