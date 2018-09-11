
// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//	 conv = Conv()
//   bn = BatchNormalization()
//
// After:
//	 bn is deleted
//   new inputs/initializers to conv are added to graph
//   any no longer used inputs/initializers are erased from graph
//
//	 this pass can handle the case satisfy all following conditions:
//	   condition 1: Run in testing mode
//     condition 2: Inputs 1 - 4 of bn are all initializer_size
//     condition 3: Output of initial conv has no other uses
//     condition 3: Currently works for only DOUBLE, FLOAT32 tensor types
//
// Formula for transformation
// $$ X_{bn} = \frac{s(X - m)}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
// $$ X_{conv} = X * W + b_{conv} $$
// thus, substituting $X$ with $X_{conv}$ in the BN equation we get:
// $$X_{bn} = X * \frac{sW}{\sqrt{\sigma + \epsilon}} + \frac{s(b_{conv} - m)}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
// or
// $$ W' = W\frac{s}{\sqrt{\sigma + \epsilon}}$$
// $$ b' = (b_{conv} - m)\frac{s}{\sqrt{\sigma + \epsilon}} + b_{bn}$$

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {
//TODO: Currently broken for complex values and float16
struct FuseBNIntoConv final : public OptimizePass {
  explicit FuseBNIntoConv()
    : OptimizePass("fuse_bn_into_conv", API_TYPE::IR) {
  }

  void replace_inputs(Tensor& W, Tensor& b, Node* conv, Graph& graph) {
    Value* new_W_value = graph.addInitializerAndInput(W);
    Value* old_W_value = conv->inputs()[1];
    conv->replaceInput(1, new_W_value);
    if (old_W_value->uses().size() == 0) {
      graph.eraseInitializerAndInput(old_W_value);
    }

    if (conv->inputs().size() == 3) {
      Value* new_b_value = graph.addInitializerAndInput(b);
      Value* old_b_value = conv->inputs()[2];
      conv->replaceInput(2, new_b_value);
      if (old_b_value->uses().size() == 0) {
        graph.eraseInitializerAndInput(old_b_value);
      }
    } else {
      Value* new_b_value = graph.addInitializerAndInput(b);
      conv->addInput(new_b_value);
    }
  }

  bool modify_conv(Node* conv, Node* bn, Graph& graph)  {
    const auto& bn_inputs = bn->inputs();
    const auto& conv_inputs = conv->inputs();
    auto end_iter = graph.initializers().end();
    auto s_iter = graph.getInitializer(bn_inputs[1]->uniqueName());
    auto bbn_iter = graph.getInitializer(bn_inputs[2]->uniqueName());
    auto m_iter = graph.getInitializer(bn_inputs[3]->uniqueName());
    auto var_iter = graph.getInitializer(bn_inputs[4]->uniqueName());
    auto W_iter = graph.getInitializer(conv_inputs[1]->uniqueName());
    if (s_iter == end_iter || bbn_iter == end_iter || m_iter == end_iter ||
        var_iter == end_iter || W_iter == end_iter) {
      return false;
    }

    ONNX_ASSERT(s_iter->sizes().size() == 1);
    ONNX_ASSERT(bbn_iter->sizes().size() == 1 && bbn_iter->sizes()[0] == s_iter->sizes()[0]);
    ONNX_ASSERT(m_iter->sizes().size() == 1 && m_iter->sizes()[0] == s_iter->sizes()[0]);
    ONNX_ASSERT(var_iter->sizes().size() == 1 && var_iter->sizes()[0] == s_iter->sizes()[0]);
    ONNX_ASSERT(W_iter->sizes().size() > 2 && W_iter->sizes()[0] == s_iter->sizes()[0]);
    ONNX_ASSERT(s_iter->elem_type() == bbn_iter->elem_type() &&
                s_iter->elem_type() == m_iter->elem_type() &&
                s_iter->elem_type() == var_iter->elem_type() &&
                s_iter->elem_type() == W_iter->elem_type());
    if (s_iter->elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
        s_iter->elem_type() != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE)  {
      return false;
    }

    Tensor bc;
    if (conv_inputs.size() == 3) {
      auto bc_iter = graph.getInitializer(conv_inputs[2]->uniqueName());
      if (bc_iter == end_iter) {
        return false;
      }
      bc = *bc_iter;
      ONNX_ASSERT(bc.sizes().size() == 1 && bc.sizes()[0] == s_iter->sizes()[0]);
    }

    Tensor s = *s_iter;
    const Tensor& bbn = *bbn_iter;
    const Tensor& m = *m_iter;
    Tensor var = *var_iter;
    Tensor W = *W_iter;
    float epsilon = bn->hasAttribute(kepsilon) ? (float) bn->f(kepsilon) : 1e-5f;
    Tensor eps;

    #define DO_COMPUTATION(TENSOR_TYPE, vec)                                   \
      eps.sizes().push_back(s.sizes()[0]);                                     \
      eps.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_##TENSOR_TYPE;    \
      for (int64_t i = 0; i < eps.sizes()[0]; ++i)  {                          \
        eps.vec().push_back(epsilon);                                          \
      }                                                                        \
      if (conv_inputs.size() != 3) {                                           \
        bc.sizes().push_back(s.sizes()[0]);                                    \
        bc.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_##TENSOR_TYPE;   \
        for (int64_t i = 0; i < eps.sizes()[0]; ++i)  {                        \
          bc.vec().push_back(0.f);                                             \
        }                                                                      \
      }                                                                        \
      var.add(eps);                                                            \
      var.sqrt();                                                              \
      s.divide(var);                                                           \
      W.scale_by_first_dim(s);                                                 \
      bc.subtract(m);                                                          \
      bc.multiply(s);                                                          \
      bc.add(bbn);                                                             \

    switch(s.elem_type()) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        DO_COMPUTATION(FLOAT, floats)
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      {
        DO_COMPUTATION(DOUBLE, doubles)
        break;
      }
      default:
        return false;
    }

    replace_inputs(W, bc, conv, graph);
    return true;
  }

  void fuse_bn_into_conv(Graph& graph) {

    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(n, [this](Graph& g){fuse_bn_into_conv(g);});
      if (n->kind() == kBatchNormalization && n->inputs()[0]->node()->kind() == kConv) {
        Node* bn = n;
        Node* conv = n->inputs()[0]->node();
        auto origInput = bn->inputs()[0];
        if (origInput->uses().size() > 1 ||
            bn->outputs().size() > 1 ||
            !modify_conv(conv, bn, graph)) {
          continue;
        }
        for (int i = 4; i >= 1; --i)  {
          if (bn->inputs()[i]->uses().size() == 1) {
            auto input = bn->inputs()[i];
            bn->removeInput(i);
            graph.eraseInitializerAndInput(input);
          }
        }
        bn->output()->replaceAllUsesWith(origInput);
        it.destroyCurrent();

      }
    }
  }

  void optimize(Graph& graph) override {
    fuse_bn_into_conv(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
