// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/defs/tensor_util.h"
#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateNopPad final : public PredicateBasedPass {
  explicit EliminateNopPad()
      : PredicateBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "eliminate_nop_pad";
  }

  static bool is_nop_pad(Node* node, Graph& graph) {
    if (node->hasAttribute(kpads)) {
      // opset 10 and below
      const auto& pads = node->is(kpads);
      for (size_t i = 0; i < pads.size(); i++) {
        // if pad constant_value is non-zero, this is not a nop pad
        if (pads[i] != 0) {
          return false;      			
        }
      }
      return true;
    } else {
      // opset 11 and above
      const auto& pads_name = node->inputs()[1]->uniqueName();
      const auto pads_initializer = graph.getInitializer(pads_name);
      // 'pad' node has the 'pads' input which has not been initialized -
      // can't proceed with elimination
      if (pads_initializer == graph.initializers().end())
        return false;

      // validate values within 'pads'
      if (pads_initializer->elem_type() == TensorProto::INT64) {
        const auto& pads = ParseData<int64_t>(&*pads_initializer);
        for (const auto& val : pads) {
          // if pad constant_value is non-zero, this is not a nop pad
          if (val != 0) {
            return false;
          }      
        }
        return true;
      }
    }

    return false;
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kPad;
  }

  bool runTransform(Node* node, Graph& graph, NodeDestroyType& destroy_current)
      override {
    if (!is_nop_pad(node, graph))
      return false;
    node->output()->replaceAllUsesWith(node->inputs()[0]);
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
