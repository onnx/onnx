
// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

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
    auto bn_inputs = bn->inputs();
    auto conv_inputs = conv->inputs();
    auto s_index = graph.get_initializer_index(bn_inputs[1]);
    auto bbn_index = graph.get_initializer_index(bn_inputs[2]);
    auto m_index = graph.get_initializer_index(bn_inputs[3]);
    auto var_index = graph.get_initializer_index(bn_inputs[4]);
    auto W_index = graph.get_initializer_index(conv_inputs[1]);
    if (s_index == -1 || bbn_index == -1 || m_index == -1 || var_index == -1 || W_index == -1) {
      return false;
    }
    auto s = graph.initializers()[s_index];
    auto bbn = graph.initializers()[bbn_index];
    auto m = graph.initializers()[m_index];
    auto var = graph.initializers()[var_index];
    auto W = graph.initializers()[W_index];
    auto epsilon = bn->hasAttribute(kepsilon) ? bn->f(kepsilon) : 1e-5;
    Tensor eps;
    Tensor bc;

    ONNX_ASSERT(s.sizes().size() == 1);
    ONNX_ASSERT(bbn.sizes().size() == 1 && bbn.sizes()[0] == s.sizes()[0]);
    ONNX_ASSERT(m.sizes().size() == 1 && m.sizes()[0] == s.sizes()[0]);
    ONNX_ASSERT(var.sizes().size() == 1 && var.sizes()[0] == s.sizes()[0]);
    ONNX_ASSERT(W.sizes().size() > 2 && W.sizes()[0] == s.sizes()[0]);

    switch(s.elem_type()) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        eps.sizes().push_back(s.sizes()[0]);
        eps.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
        for (int64_t i = 0; i < eps.sizes()[0]; i++)  {
          eps.floats().push_back(epsilon);
        }
        bc.sizes().push_back(s.sizes()[0]);
        bc.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
        for (int64_t i = 0; i < eps.sizes()[0]; i++)  {
          bc.floats().push_back(0.f);
        }

        if (conv_inputs.size() == 3) {
          auto bc_index = graph.get_initializer_index(conv_inputs[2]);
          if (bc_index == -1) {
            return false;
          }
          bc = graph.initializers()[bc_index];
          ONNX_ASSERT(bc.sizes().size() == 1 && bc.sizes()[0] == s.sizes()[0]);
        }

        var.add(eps);
        var.sqrt();
        s.divide(var);
        if (!s.is_raw_data()) {
          const char * ptr = reinterpret_cast<const char *>(s.floats().data());
          std::string string_rep(ptr, ptr + sizeof(float) * s.floats().size());
          s.set_raw_data(string_rep);
        }

        W.scale_by_first_dim(s);
        bc.subtract(m);
        bc.multiply(s);
        bc.add(bbn);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      {
        eps.sizes().push_back(s.sizes()[0]);
        eps.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
        for (int64_t i = 0; i < eps.sizes()[0]; i++)  {
          eps.doubles().push_back((double)epsilon);
        }
        bc.sizes().push_back(s.sizes()[0]);
        bc.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
        for (int64_t i = 0; i < eps.sizes()[0]; i++)  {
          bc.doubles().push_back(0.);
        }

        if (conv_inputs.size() == 3) {
          auto bc_index = graph.get_initializer_index(conv_inputs[2]);
          if (bc_index == -1) {
            return false;
          }
          bc = graph.initializers()[bc_index];
          ONNX_ASSERT(bc.sizes().size() == 1 && bc.sizes()[0] == s.sizes()[0]);
        }

        var.add(eps);
        var.sqrt();
        s.divide(var);
        if (!s.is_raw_data()) {
          const char * ptr = reinterpret_cast<const char *>(s.doubles().data());
          std::string string_rep(ptr, ptr + sizeof(double) * s.doubles().size());
          s.set_raw_data(string_rep);
        }
        W.scale_by_first_dim(s);
        bc.subtract(m);
        bc.multiply(s);
        bc.add(bbn);
        break;
      }
      default:
        throw("Incompatible data type");
    }

    replace_inputs(W, bc, conv, graph);
    return true;
  }

  void fuse_bn_into_conv(Graph& graph) {

    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(n, [this](Graph& g){fuse_bn_into_conv(g);});
      if (n->kind() == kBatchNormalization && n->inputs()[0]->node()->kind() == kConv) {
        auto origInput = n->inputs()[0];
        if (origInput->uses().size() > 1 ||
            n->outputs().size() > 1 ||
            !modify_conv(origInput->node(), n, graph)) {
          continue;
        }
        for (int i = 1; i <= 4; i++)  {
          if (n->inputs()[i]->uses().size() == 0) {
            graph.eraseInitializerAndInput(n->inputs()[i]);
          }
        }
        n->output()->replaceAllUsesWith(origInput);
        it.destroyCurrent();
      }
    }
  }

  void optimize(Graph& graph) override {
    fuse_bn_into_conv(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
