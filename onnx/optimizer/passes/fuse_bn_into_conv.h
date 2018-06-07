
// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"
#include <math.h>

namespace ONNX_NAMESPACE { namespace optimization {
//TODO: Currently broken for complex values and float16
struct FuseBNIntoConv final : public OptimizePass {
  explicit FuseBNIntoConv()
    : OptimizePass("fuse_consecutive_transposes", API_TYPE::IR) {
  }

  template<typename T>
  static void add_nums(void* x, const void* y) {
    T* x_cast = (T*) x;
    const T* y_cast = (const T*) y;
    *x_cast = *x_cast + *y_cast;
  }

  template<typename T>
  static void sub_nums(void* x, const void* y) {
    T* x_cast = (T*) x;
    const T* y_cast = (const T*) y;
    *x_cast = *x_cast - *y_cast;
  }

  template<typename T>
  static void mult_nums(void* x, const void* y) {
    T* x_cast = (T*) x;
    const T* y_cast = (const T*) y;
    *x_cast = *x_cast * *y_cast;
  }

  template<typename T>
  static void divide_nums(void* x, const void* y) {
    T* x_cast = (T*) x;
    const T* y_cast = (const T*) y;
    *x_cast = *x_cast / *y_cast;
  }
  template<typename T>
  static void sqrt_num(void* x) {
    T* x_cast = (T*) x;
    *x_cast = (T) sqrt((double) *x_cast);
  }

  void replace_inputs(Tensor& W, Tensor& b, Node* conv, Graph& graph) {
    Value* new_W_value = graph.addInitializerAndInput(W);
    Value* old_W_value = conv->inputs()[1];
    conv->replaceInput(1, new_W_value);
    graph.erase_old_initializer(old_W_value);
    Value* new_b_value = graph.addInitializerAndInput(b);
    if (conv->inputs().size() == 3) {
      Value* old_b_value = conv->inputs()[2];
      conv->replaceInput(2, new_b_value);
      graph.erase_old_initializer(old_b_value);
    } else {
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
    auto epsilon = bn->f(kepsilon);
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
        for (int i = 0; i < eps.sizes()[0]; i++)  {
          eps.floats().push_back(epsilon);
        }
        bc.sizes().push_back(s.sizes()[0]);
        bc.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
        for (int i = 0; i < eps.sizes()[0]; i++)  {
          bc.floats().push_back(0.f);
        }

        if (conv_inputs.size() == 3) {
          auto bc_index = graph.get_initializer_index(conv_inputs[1]);
          if (bc_index == -1) {
            return false;
          }
          bc = graph.initializers()[bc_index];
          ONNX_ASSERT(bc.sizes() == 1 && bc.sizes()[0] == s.sizes()[0]);
        }

        var.apply_binary_function(&(add_nums<float>), eps);
        var.apply_unary_function(&(sqrt_num<float>));
        s.apply_binary_function(&(divide_nums<float>), var);
        if (!s.is_raw_data()) {
          const char * ptr = reinterpret_cast<const char *>(s.floats().data());
          std::string string_rep(ptr, ptr + sizeof(float) * s.floats().size());
          s.set_raw_data(string_rep);
        }
        W.scale_by_first_dim(s);
        bc.apply_binary_function(&(sub_nums<float>), m);
        bc.apply_binary_function(&(mult_nums<float>), s);
        bc.apply_binary_function(&(add_nums<float>), bbn);
        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      {
        eps.sizes().push_back(s.sizes()[0]);
        eps.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
        for (int i = 0; i < eps.sizes()[0]; i++)  {
          eps.doubles().push_back((double)epsilon);
        }
        bc.sizes().push_back(s.sizes()[0]);
        bc.elem_type() = ONNX_NAMESPACE::TensorProto_DataType_DOUBLE;
        for (int i = 0; i < eps.sizes()[0]; i++)  {
          bc.doubles().push_back(0.);
        }

        if (conv_inputs.size() == 3) {
          auto bc_index = graph.get_initializer_index(conv_inputs[1]);
          if (bc_index == -1) {
            return false;
          }
          bc = graph.initializers()[bc_index];
          ONNX_ASSERT(bc.sizes() == 1 && bc.sizes()[0] == s.sizes()[0]);
        }

        var.apply_binary_function(&(add_nums<double>), eps);
        var.apply_unary_function(&(sqrt_num<double>));
        s.apply_binary_function(&(divide_nums<double>), var);
        if (!s.is_raw_data()) {
          char const * ptr = reinterpret_cast<char const *>(s.doubles().data());
          std::string string_rep(ptr, ptr + sizeof(double) * s.doubles().size());
          s.set_raw_data(string_rep);
        }
        W.scale_by_first_dim(s);
        bc.apply_binary_function(&(sub_nums<double>), m);
        bc.apply_binary_function(&(mult_nums<double>), s);
        bc.apply_binary_function(&(add_nums<double>), bbn);
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
            !modify_conv(n->input()->node(), n, graph)) {
          continue;
        }
        for (int i = 1; i <=4; i++)  {
          graph.erase_old_initializer(n->inputs()[i]);
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
