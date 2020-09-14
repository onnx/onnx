// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   X is a tensor with shape=[1, 1, 2, 3, 1, 5, 1]
//   Y = Squeeze(X, axes=[1, 4]) -> shape=[1, 2, 3, 5, 1]
//   Z = Squeeze(Y, axes=[0, 4]) -> shape=[2, 3, 5]
// After:
//   Z = Squeeze(X, axes=[0, 1, 4, 6])
#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseConsecutiveSqueezes final : public PredicateBasedPass {
  explicit FuseConsecutiveSqueezes()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "fuse_consecutive_squeezes";
  }
  // returns a vector `ret` such that squeeze by `ret` is equivalent
  // to squeeze by `axes_1` and then by `axes_2`
  std::vector<int64_t> compose_squeezes(
      const std::vector<int64_t>& axes_1,
      const std::vector<int64_t>& axes_2) {
    std::vector<int64_t> ret;
    ret.reserve(axes_1.size() + axes_2.size());

    std::vector<int64_t> sorted_axes_1(axes_1.begin(), axes_1.end());
    std::sort(sorted_axes_1.begin(), sorted_axes_1.end());
    std::copy(
        sorted_axes_1.begin(), sorted_axes_1.end(), std::back_inserter(ret));

    for (int64_t i : axes_2) {
      for (auto iter = sorted_axes_1.begin(); iter != sorted_axes_1.end();
           ++iter) {
        // if current axis 1 - prev_num is bigger than axis 2
        // put axis 2 + prev_num as new axis
        int64_t prev_num = std::distance(sorted_axes_1.begin(), iter);
        if (*iter - prev_num > i) {
          ret.push_back(i + prev_num);
          break;
        }
        // if no current axis 1 - prev_num is bigger than axis 2
        // put axis 2 + prev_num + 1 as new axis
        if (std::next(iter) == sorted_axes_1.end()) {
          ret.push_back(i + prev_num + 1);
        }
      }
    }
    std::sort(ret.begin(), ret.end());
    return ret;
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kSqueeze &&
        node->inputs()[0]->node()->kind() == kSqueeze;
  }
  bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current)
      override {
    auto orig_input = n->inputs()[0];

    // Read axes 1
    auto axes1_input = orig_input->node()->inputs()[1];
    auto axes1_initializer = graph.getInitializer(axes1_input->uniqueName());
    if (axes1_initializer == graph.initializers().end())
      return false;
    const auto& axes1 = ParseData<int64_t>(&*axes1_initializer);
    //Read axes 2
    auto axes2_input = n->inputs()[1];
    auto axes2_initializer = graph.getInitializer(axes2_input->uniqueName());
    if (axes2_initializer == graph.initializers().end())
      return false;
    const auto& axes2 = ParseData<int64_t>(&*axes2_initializer);
    auto combined_axes = compose_squeezes(axes1, axes2);

    Tensor t;
    t.elem_type() = TensorProto_DataType_INT64;
    t.sizes() = std::vector<int64_t>{static_cast<int64_t>(combined_axes.size())};
    auto& data = t.int64s();
    for (auto a : combined_axes) {
      data.emplace_back(a);
    }
    Value* new_axes_value = graph.addInitializerAndInput(t);
    n->replaceInput(0, orig_input->node()->inputs()[0]);
    n->replaceInput(1, new_axes_value);
    if (orig_input->uses().size() == 0) {
      orig_input->node()->destroy();
    }
    if (axes1_input->uses().size() == 0) {
      graph.eraseInitializerAndInput(axes1_input);
    }
    if (axes2_input->uses().size() == 0) {
      graph.eraseInitializerAndInput(axes2_input);
    }
    destroy_current = NodeDestroyType::DestroyZero;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
