// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   Z = MatMul(X, Y)
//   A = Z + Bias
// After:
//   A = Gemm(X, Y, Bias)
//
// the pass can handle the case when:
//   case 1: Bias is 1D tensor and Bias.dim[0] == Z.dim[1]
//   case 2: Bias is 2D tensor and Bias.dim[0] == Z.dim[0] or 1
//           and Bias.dim[1] = Z.dim[1]

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
        node->inputs()[0]->node()->kind() == kMatMul;
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
    auto x_shape = orig_matmul->node()->inputs()[0]->sizes();
    auto y_shape = orig_matmul->node()->inputs()[1]->sizes();
    int64_t z_N = -1;
    int64_t z_M = -1;
    // try to get feature N from x_shape
    if (static_cast<int64_t>(x_shape.size()) == 2 && x_shape[0].is_int) {
      z_N = x_shape[0].dim;
    } else {
      return false;
    }
    // try to get feature M from y_shape
    if (static_cast<int64_t>(y_shape.size()) == 2 && y_shape[1].is_int) {
      z_M = y_shape[1].dim;
    } else {
      return false;
    }
    // check if bias_shape is compatible
    auto bias_shape = orig_bias->sizes();
    auto bias_dim = static_cast<int64_t>(bias_shape.size());
    int64_t bias_N = -1;
    int64_t bias_M = -1;
    if (bias_dim == 1 && bias_shape[0].is_int) {
      bias_N = 1;
      bias_M = bias_shape[0].dim;
    } else if (bias_dim == 2 && bias_shape[0].is_int && bias_shape[1].is_int) {
      bias_N = bias_shape[0].dim;
      bias_M = bias_shape[1].dim;
    } else {
      return false;
    }
    if ((bias_N != z_N && bias_N != 1) || bias_M != z_M) {
        return false;
    }
    // proceed to fuse MatMul and Add into Gemm
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
