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
//     C = Squeeze(A, ...)
//     B = Add(bias, C)
//     Y = BatchNorm(X, scale, B, mean, var)
//   case 2:
//     C = Squeeze(A, ...)
//     B = Sub(bias, C)
//     Y = BatchNorm(X, scale, B, mean, var)
//   case 3:
//     C = Squeeze(A, ...)
//     S = Mul(scale, C)
//     B = Mul(bias, C)
//     Y = BatchNorm(X, S, B, mean, var)
//   case 4:
//     C = Squeeze(A, ...)
//     S = Div(scale, C)
//     B = Div(bias, C)
//     Y = BatchNorm(X, S, B, mean, var)
//   and there is no Z in graph.

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseArithmeticIntoBatchNorm final : public OptimizePass {
  explicit FuseArithmeticIntoBatchNorm()
      : OptimizePass("fuse_arithmetic_into_batch_norm", API_TYPE::IR) {}

  void fuse_arithmetic_into_batch_norm(Graph& graph) {
    int size_lack_count = 0;
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(
          n, [this](Graph& g) { fuse_arithmetic_into_batch_norm(g); });

      // target kinds of arithmetic node
      std::vector<NodeKind> kind_group = {kAdd, kSub, kMul, kDiv};
      NodeKind node_kind = n->kind();
      if (std::find(kind_group.begin(), kind_group.end(), node_kind) ==
          kind_group.end()) {
        continue;
      }

      // for Sub and Div, first input should be BatchNormalization
      if ((node_kind == kSub || node_kind == kDiv) &&
          n->inputs()[0]->node()->kind() != kBatchNormalization) {
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

      // get BatchNormalization shape
      auto bn_shape = orig_batch_norm->sizes();
      if (bn_shape.size() == 0) {
        bn_shape = orig_batch_norm->node()->inputs()[0]->sizes();
      }
      if (bn_shape.size() == 0) {
        // no enough information, bail out
        size_lack_count++;
        continue;
      }

      Value* orig_scale = orig_batch_norm->node()->inputs()[1];
      Value* orig_bias = orig_batch_norm->node()->inputs()[2];

      // get num of Channels C
      int64_t C = -1;
      if (bn_shape.size() > 1 && bn_shape[1].is_int) {
        C = bn_shape[1].dim;
      } else if (
          orig_scale->sizes().size() == 1 && orig_scale->sizes()[0].is_int) {
        C = orig_scale->sizes()[0].dim;
      } else if (
          orig_bias->sizes().size() == 1 && orig_scale->sizes()[0].is_int) {
        C = orig_bias->sizes()[0].dim;
      }

      auto const_shape = orig_const->sizes();

      // arithmetic param dim should not be greater than batch norm dim
      if (const_shape.size() > bn_shape.size()) {
        continue;
      }

      // check if is valid broadcast
      // only dim is C or 1 is acceptable
      // e.g. for 4-dims input, const dims should be (1, C, 1, 1), (C, 1, 1)
      // or (1, 1, 1, 1), (1, 1, 1)
      int64_t num_el = 1;
      for (int i = 0; i < static_cast<int64_t>(const_shape.size()); ++i) {
        if (const_shape[i].is_int &&
            (const_shape[i].dim == C || const_shape[i].dim == 1)) {
          num_el *= const_shape[i].dim;
          // C should be in index 1
          if (const_shape[i].dim == C &&
              i != 1 + const_shape.size() - bn_shape.size()) {
            num_el = -1;
            break;
          }
        } else {
          num_el = -1;
          break;
        }
      }
      if (num_el == -1 || C == -1) {
        size_lack_count++;
        continue;
      }

      std::vector<int64_t> squeeze_axes(const_shape.size());
      std::iota(squeeze_axes.begin(), squeeze_axes.end(), 0);
      if (num_el == 1 || num_el == C) {
        squeeze_axes.erase(
            squeeze_axes.begin() + 1 + const_shape.size() - bn_shape.size());
      } else {
        continue;
      }

      // add squeeze node for const
      Node* squeeze = graph.create(kSqueeze, 1);
      squeeze->addInput(orig_const);
      squeeze->is_(kaxes, std::move(squeeze_axes));
      std::vector<Dimension> squeezed_dims = {num_el};
      squeeze->output()->setSizes(squeezed_dims);
      squeeze->insertBefore(orig_batch_norm->node());

      std::vector<Dimension> dim_c = {C};
      // add arithmetic node for scale
      if (node_kind == kMul || node_kind == kDiv) {
        Node* arith_scale = graph.create(node_kind, 1);
        arith_scale->addInput(orig_scale);
        arith_scale->addInput(squeeze->output());
        arith_scale->insertBefore(orig_batch_norm->node());
        arith_scale->output()->setSizes(dim_c);
        orig_batch_norm->node()->replaceInput(1, arith_scale->output());
      }

      // add arithmetic node for bias
      Node* arith_bias = graph.create(node_kind, 1);
      arith_bias->addInput(orig_bias);
      arith_bias->addInput(squeeze->output());
      arith_bias->insertBefore(orig_batch_norm->node());
      arith_bias->output()->setSizes(dim_c);
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
    if (size_lack_count != 0) {
      std::cout << "Warning: failed to fuse arithmetic into BatchNormalization "
                   "due to lack of size information."
                << std::endl;
    }
  }

  void optimize(Graph& graph) override {
    fuse_arithmetic_into_batch_norm(graph);
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
