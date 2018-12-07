// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   Z = Conv(X, Y)
//   B = Z + A
// After:
//   B = Conv(X, Y, A)
//
// the pass can handle the following cases:
//   case 1: A is 1D tensor and A.dim[0] == Z.dim[1]
//   case 2: A is 1-element 1D tensor

#include <numeric>

#include "onnx/common/assertions.h"
#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseAddBiasIntoConv final : public PredicateBasedPass {
  explicit FuseAddBiasIntoConv()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_add_bias_into_conv";
  }
  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kAdd && node->inputs()[0]->node()->kind() == kConv &&
        node->inputs()[0]->node()->inputs().size() == 2;
  }
  bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current)
      override {
    // due to current broadcasting's constraint, Conv has to be the first
    // operand
    destroy_current = NodeDestroyType::DestroyZero;
    auto orig_conv = n->inputs()[0];
    auto orig_bias = n->inputs()[1];
    // check if bias is Const or in graph's initializers
    if (orig_bias->node()->kind() != kConstant &&
        orig_bias->node()->kind() != kParam) {
      return false;
    }
    // check if conv is only used by Add
    if (orig_conv->uses().size() > 1) {
      return false;
    }
    auto conv_shape = orig_conv->sizes();
    auto bias_shape = orig_bias->sizes();
    auto weight_shape = orig_conv->node()->inputs()[1]->sizes();
    int64_t M = -1;
    int64_t rank = -1;
    // try to get feature M and rank from conv_shape
    if (conv_shape.size() > 1 && conv_shape[1].is_int) {
      M = conv_shape[1].dim;
      rank = conv_shape.size();
    }
    // try to get feature M and rank from weight_shape
    if (weight_shape.size() > 0 && weight_shape[0].is_int) {
      ONNX_ASSERT(M == -1 || M == weight_shape[0].dim);
      M = weight_shape[0].dim;
      ONNX_ASSERT(
          rank == -1 || rank == static_cast<int64_t>(weight_shape.size()));
      rank = weight_shape.size();
    }
    int64_t num_el = 1;
    for (int i = 0; i < static_cast<int64_t>(bias_shape.size()); ++i) {
      if (bias_shape[i].is_int) {
        num_el *= bias_shape[i].dim;
      } else {
        num_el = -1;
        return false;
      }
    }
    if (M == -1 || num_el == -1) {
      // No enough information, bail out
      return false;
    }
    if (rank < static_cast<int64_t>(bias_shape.size())) {
      return false;
    }
    if (num_el == 1) {
      if (orig_bias->node()->kind() != kParam &&
          orig_conv->node()->isBefore(orig_bias->node())) {
        orig_bias->node()->moveBefore(orig_conv->node());
      }
      Value* conv_3rd_input = orig_bias;
      if (bias_shape.size() > 1) {
        Node* squeeze = graph.create(kSqueeze, 1);
        std::vector<int64_t> axes(bias_shape.size() - 1);
        std::iota(axes.begin(), axes.end(), 0);
        squeeze->is_(kaxes, std::move(axes));
        squeeze->addInput(conv_3rd_input);
        conv_3rd_input = squeeze->output();
        squeeze->insertBefore(orig_conv->node());
      }
      if (M > 1) {
        Node* constant = graph.create(kConstant, 1);
        Tensor t;
        t.sizes().push_back(static_cast<int64_t>(1));
        t.int64s().push_back(M);
        t.elem_type() = TensorProto_DataType_INT64;
        Symbol sym = Symbol("value");
        constant->t_(sym, t);
        std::vector<Dimension> s = {1};
        constant->output()->setSizes(s);
        constant->output()->setElemType(TensorProto_DataType_INT64);
        constant->insertBefore(orig_conv->node());
        Node* tile = graph.create(kTile, 1);
        tile->addInput(conv_3rd_input);
        tile->addInput(constant->output());
        conv_3rd_input = tile->output();
        tile->insertBefore(orig_conv->node());
      }
      orig_conv->node()->addInput(conv_3rd_input);
    } else if (rank > static_cast<int64_t>(bias_shape.size()) + 1) {
      return false;
    } else if (
        num_el == M &&
        bias_shape[1 + bias_shape.size() - static_cast<unsigned>(rank)].dim ==
            M) {
      ONNX_ASSERT(bias_shape.size() > 1);
      if (orig_bias->node()->kind() != kParam &&
          orig_conv->node()->isBefore(orig_bias->node())) {
        orig_bias->node()->moveBefore(orig_conv->node());
      }
      Node* squeeze = graph.create(kSqueeze, 1);
      std::vector<int64_t> axes(bias_shape.size());
      std::iota(axes.begin(), axes.end(), static_cast<int64_t>(0));
      axes.erase(
          axes.begin() + 1 + bias_shape.size() - static_cast<unsigned>(rank));
      squeeze->is_(kaxes, std::move(axes));
      squeeze->addInput(orig_bias);
      squeeze->insertBefore(orig_conv->node());
      orig_conv->node()->addInput(squeeze->output());
    } else {
      return false;
    }
    if (orig_conv->sizes().size() == 0 && n->output()->sizes().size() > 0) {
      orig_conv->setSizes(n->output()->sizes());
    }
    if (n->output()->elemType() != TensorProto_DataType_UNDEFINED) {
      orig_conv->setElemType(n->output()->elemType());
    }
    n->replaceAllUsesWith(orig_conv->node());
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
