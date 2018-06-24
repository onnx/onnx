
// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   bn = BatchNormalization()
//   mul = Mul() with channel broadcast
//   add = Add() with channel broadcast
//
// After:
//   mul and add are deleted
//   update scale and bias
//
// this pass can handle the case satisfy all following conditions:
// condition 1: Run in testing mode
// condition 2: Output of bn has no other uses
//
// use case: caffe represent BatchNorm Layer by BatchNorm + Scale (Mul + Add)
//           so this optimization is useful when the model is translated from
//           caffe -> caffe2 -> onnx
//
// Formula for transformation
//
// (scale * Norm + bias) * mul_op2 + add_op2
//
// new scale = scale * mul_op2
// new bias = bias * mul_op2 + add_op2

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {
struct FuseMulAddIntoBN final : public OptimizePass {
  explicit FuseMulAddIntoBN()
      : OptimizePass("fuse_mul_add_into_bn", API_TYPE::IR) {}

  void replace_scale_input(Tensor& scale, Node* bn, Graph& graph) {
    Value* new_scale_value = graph.addInitializerAndInput(scale);
    Value* old_scale_value = bn->inputs()[1];
    bn->replaceInput(1, new_scale_value);
    if (old_scale_value->uses().size() == 0) {
      graph.eraseInitializerAndInput(old_scale_value);
    }
  }

  void replace_bias_input(Tensor& bias, Node* bn, Graph& graph) {
    Value* new_bias_value = graph.addInitializerAndInput(bias);
    Value* old_bias_value = bn->inputs()[2];
    bn->replaceInput(2, new_bias_value);
    if (old_bias_value->uses().size() == 0) {
      graph.eraseInitializerAndInput(old_bias_value);
    }
  }

  bool fuse_mul_into_bn(Node* bn, Node* mul, Graph& graph) {
    const auto& bn_inputs = bn->inputs();
    const auto& mul_inputs = mul->inputs();
    auto end_iter = graph.initializers().end();
    auto scale_iter = graph.getInitializer(bn_inputs[1]->uniqueName());
    auto bias_iter = graph.getInitializer(bn_inputs[2]->uniqueName());
    auto mul_iter = graph.getInitializer(mul_inputs[1]->uniqueName());
    if (scale_iter == end_iter || bias_iter == end_iter ||
        mul_iter == end_iter) {
      return false;
    }

    Tensor scale = *scale_iter;
    Tensor bias = *bias_iter;
    Tensor mul_tensor = *mul_iter;
    scale.multiply(mul_tensor);
    bias.multiply(mul_tensor);

    replace_scale_input(scale, bn, graph);
    replace_bias_input(bias, bn, graph);
    if (mul_inputs[1]->uses().size() == 1) {
      auto input = mul_inputs[1];
      mul->removeInput(1);
      graph.eraseInitializerAndInput(input);
    }
    mul->output()->replaceAllUsesWith(mul_inputs[0]);
    return true;
  }

  bool fuse_add_into_bn(Node* bn, Node* add, Graph& graph) {
    const auto& bn_inputs = bn->inputs();
    const auto& add_inputs = add->inputs();
    auto end_iter = graph.initializers().end();
    auto scale_iter = graph.getInitializer(bn_inputs[1]->uniqueName());
    auto bias_iter = graph.getInitializer(bn_inputs[2]->uniqueName());
    auto add_iter = graph.getInitializer(add_inputs[1]->uniqueName());
    if (scale_iter == end_iter || bias_iter == end_iter ||
        add_iter == end_iter) {
      return false;
    }

    Tensor bias = *bias_iter;
    Tensor add_tensor = *add_iter;
    bias.add(add_tensor);

    replace_bias_input(bias, bn, graph);
    if (add_inputs[1]->uses().size() == 1) {
      auto input = add_inputs[1];
      add->removeInput(1);
      graph.eraseInitializerAndInput(input);
    }
    add->output()->replaceAllUsesWith(add_inputs[0]);
    return true;
  }

  void fuse_mul_add_into_bn(Graph& graph) {
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(
          n, [this](Graph& g) { fuse_mul_add_into_bn(g); });
      if (n->kind() == kMul &&
          n->inputs()[0]->node()->kind() == kBatchNormalization) {
        auto* bn = n->inputs()[0]->node();
        if (bn->outputs()[0]->uses().size() != 1)
          continue;
        if (!n->hasAttribute(Symbol("axis")) || n->i(Symbol("axis")) != 1)
          continue;
        if (!n->hasAttribute(Symbol("broadcast")) ||
            n->i(Symbol("broadcast")) != 1)
          continue;
        if (fuse_mul_into_bn(bn, n, graph)) {
          it.destroyCurrent();
        }
      }
      if (n->kind() == kAdd &&
          n->inputs()[0]->node()->kind() == kBatchNormalization) {
        auto* bn = n->inputs()[0]->node();
        if (bn->outputs()[0]->uses().size() != 1)
          continue;
        if (!n->hasAttribute(Symbol("axis")) || n->i(Symbol("axis")) != 1)
          continue;
        if (!n->hasAttribute(Symbol("broadcast")) ||
            n->i(Symbol("broadcast")) != 1)
          continue;
        if (fuse_add_into_bn(bn, n, graph))
          it.destroyCurrent();
      }
    }
  }

  void optimize(Graph& graph) override {
    fuse_mul_add_into_bn(graph);
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
