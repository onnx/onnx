// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConsecutiveSqueezes final : public OptimizePass {
  explicit FuseConsecutiveSqueezes()
      : OptimizePass("fuse_consecutive_squeezes", API_TYPE::IR) {}

  // returns a vector `ret` such that squeeze by `ret` is equivalent
  // to squeeze by `t1` and then by `t2`
  std::vector<int64_t> compose_squeezes(
      const std::vector<int64_t>& t1,
      const std::vector<int64_t>& t2) {
    std::vector<int64_t> ret;
    ret.reserve(t1.size() + t2.size());
    for (auto i : t1) {
      ret.push_back(i);
    }
    for (auto i : t2) {
      ret.push_back(i + t1.size());
    }
    return ret;
  }

  void fuse_consecutive_squeezes(Graph& graph) {
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(
          n, [this](Graph& g) { fuse_consecutive_squeezes(g); });
      if (n->kind() == kSqueeze && n->input()->node()->kind() == kSqueeze) {
        auto orig_input = n->input();
        n->is_(
            kaxes,
            compose_squeezes(orig_input->node()->is(kaxes), n->is(kaxes)));
        n->replaceInput(0, orig_input->node()->input());
        if (orig_input->uses().size() == 0) {
          orig_input->node()->destroy();
        }
      }
    }
  }

  void optimize(Graph& graph) override {
    fuse_consecutive_squeezes(graph);
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
