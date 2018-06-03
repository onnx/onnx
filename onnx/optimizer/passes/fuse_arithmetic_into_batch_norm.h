// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   case 1:
//     Y = BatchNorm(X, scale, bias, mean, var)
//     Z = Add(Y, A)
//   case 2:
//     Y = BatchNorm(X, scale, bias, mean, var)
//     Z = Sub(Y, A)
//   case 1:
//     Y = BatchNorm(X, scale, bias, mean, var)
//     Z = Mul(Y, A)
//   case 2:
//     Y = BatchNorm(X, scale, bias, mean, var)
//     Z = Div(Y, A)
// After:
//   case 1:
//     B = Add(bias, A)
//     Y = BatchNorm(X, scale, B, mean, var)
//   case 2:
//     B = Sub(bias, A)
//     Y = BatchNorm(X, scale, B, mean, var)
//   case 3:
//     S = Mul(scale, A)
//     B = Mul(bias, A)
//     Y = BatchNorm(X, S, B, mean, var)
//   case 4:
//     S = Div(scale, A)
//     B = Div(bias, A)
//     Y = BatchNorm(X, S, B, mean, var)
//   and there is no Z in graph.

#include <numeric>

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseArithmeticIntoBatchNorm final : public OptimizePass {
  explicit FuseArithmeticIntoBatchNorm()
      : OptimizePass("fuse_arithmetic_into_batch_norm", API_TYPE::IR) {}

  void fuse_arithmetic_into_batch_norm(Graph& graph) {
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(
          n, [this](Graph& g) { fuse_arithmetic_into_batch_norm(g); });

      // target kinds of arithmetic node
      std::vector<NodeKind> kind_group = {kAdd, kSub, kMul, kDiv};
      NodeKind node_kind = n->kind();
      if (std::find(kind_group.begin(), kind_group.end(), n->kind()) ==
          kind_group.end()) {
        continue;
      }

      Value* orig_batch_norm;
      Value* orig_const;

      for (auto* input : n->inputs()) {
        auto kind = input->node()->kind();
        if (kind == kParam || kind == kConstant) {
          orig_const = input;
        }
        if (kind == kBatchNormalization) {
          orig_batch_norm = input;
        }
      }

      // check if inputs are not one Const or initializer
      // and one BatchNormalization
      if (!orig_batch_norm || !orig_const) {
        continue;
      }

      // check if BatchNorm is only used by one node
      bool bn_multi_use = false;
      for (auto output : orig_batch_norm->node()->outputs()) {
        if (output->uses().size() > 1) {
          bn_multi_use = true;
          break;
        }
      }
      if (bn_multi_use) {
        continue;
      }

      // add arithmetic node for scale
      if (node_kind == kMul || node_kind == kDiv) {
        Node* arith_scale = graph.create(node_kind, 1);
        arith_scale->addInput(orig_batch_norm->node()->inputs()[1]);
        arith_scale->addInput(orig_const);
        arith_scale->insertBefore(orig_batch_norm->node());
        orig_batch_norm->node()->replaceInput(1, arith_scale->output());
      }

      // add arithmetic node for bias
      Node* arith_bias = graph.create(node_kind, 1);
      arith_bias->addInput(orig_batch_norm->node()->inputs()[2]);
      arith_bias->addInput(orig_const);
      arith_bias->insertBefore(orig_batch_norm->node());
      orig_batch_norm->node()->replaceInput(2, arith_bias->output());

      if (orig_batch_norm->sizes().size() == 0 &&
          n->output()->sizes().size() > 0) {
        orig_batch_norm->setSizes(n->output()->sizes());
      }
      if (n->output()->elemType() != TensorProto_DataType_UNDEFINED) {
        orig_batch_norm->setElemType(n->output()->elemType());
      }
      n->replaceAllUsesWith(orig_batch_norm->node());
      it.destroyCurrent();
    }
  }

  void optimize(Graph& graph) override {
    fuse_arithmetic_into_batch_norm(graph);
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE