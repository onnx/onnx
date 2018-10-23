// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   Z = MatMul(X, Y)
//   B = Z + A
// After:
//   B = Gemm(X, Y, A)
//
// the pass can handle the case when A is 1D tensor and A.dim[0] == Z.dim[1]

#include <numeric>

#include "onnx/common/assertions.h"
#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseMatMulAddBiasIntoGemm final : public PredicateBasedPass {
  explicit FuseMatMulAddBiasIntoGemm()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_matmul_add_bias_into_gemm";
  }
  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kAdd &&
        strcmp(node->inputs()[0]->node()->kind().toString(), "MatMul") == 0 &&
        node->inputs()[0]->node()->inputs().size() == 2;
  }
  bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current)
      override {
    // due to current broadcasting's constraint, MatMul has to be the first
    // operand
    destroy_current = NodeDestroyType::DestroyZero;
    auto orig_matmul = n->inputs()[0];
    auto orig_bias = n->inputs()[1];
    // check if bias is Const or in graph's initializers
    if (orig_bias->node()->kind() != kConstant &&
        orig_bias->node()->kind() != kParam) {
      return false;
    }
    // check if MatMul is only used by Add
    if (orig_matmul->uses().size() > 1) {
      return false;
    }
    auto bias_shape = orig_bias->sizes();
    auto weight_shape = orig_matmul->node()->inputs()[1]->sizes();
    int64_t M = -1;
    // try to get feature M from weight_shape
    if (static_cast<int64_t>(weight_shape.size()) == 2 &&
        weight_shape[1].is_int) {
      M = weight_shape[1].dim;
    } else {
      return false;
    }
    // check if bias_shape is compatible
    int64_t num_el = 1;
    for (int i = 0; i < static_cast<int64_t>(bias_shape.size()); ++i) {
      if (bias_shape[i].is_int) {
        num_el *= bias_shape[i].dim;
      } else {
        num_el = -1;
        return false;
      }
    }
    if (num_el != M || bias_shape.back().dim != M) {
      return false;
    }
    Node* gemm = graph.create(kGemm,
        orig_matmul->node()->inputs(),
        n->outputs().size());
    gemm->addInput(n->inputs()[1]);
    for (int i = 0; i < static_cast<int64_t>(gemm->outputs().size()); ++i) {
      gemm->outputs()[i]->copyMetadata(n->outputs()[i]);
    }
    gemm->f_(kalpha, 1.0);
    gemm->f_(kbeta, 1.0);
    gemm->i_(ktransA, 0);
    gemm->i_(ktransB, 0);
    gemm->insertBefore(orig_matmul->node());
    n->replaceAllUsesWith(gemm);
    destroy_current = NodeDestroyType::DestroyTwo;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
