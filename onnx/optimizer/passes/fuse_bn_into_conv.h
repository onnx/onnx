
// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"
#include <math.h>

namespace ONNX_NAMESPACE { namespace optimization {

struct FuseBNIntoConv final : public OptimizePass {
  explicit FuseBNIntoConv()
    : OptimizePass("fuse_consecutive_transposes", API_TYPE::IR) {
  }

  template<typename T>
  void add_nums(void* x, void* y) {
    T* x_cast = (T*) x;
    T* y_cast = (T*) y;
    *x_cast = *x_cast + *y_cast;
  }

  template<typename T>
  void sub_nums(void* x, void* y) {
    T* x_cast = (T*) x;
    T* y_cast = (T*) y;
    *x_cast = *x_cast - *y_cast;
  }

  template<typename T>
  void mult_nums(void* x, void* y) {
    T* x_cast = (T*) x;
    T* y_cast = (T*) y;
    *x_cast = *x_cast * *y_cast;
  }

  template<typename T>
  void divide_nums(void* x, void* y) {
    T* x_cast = (T*) x;
    T* y_cast = (T*) y;
    *x_cast = *x_cast / *y_cast;
  }
  template<typename T>
  void sqrt_num(void* x) {
    T* x_cast = (T*) x;
    *x_cast = (T) sqrt((double) *x_cast);
  }

  void handle_old_initializer(Value* v, Graph& graph) {
    if (v.uses().size() == 0) {
      graph.eraseInitializer(v->uniqueName());
      graph.freeValue(v);
    }
  }

  void replace_inputs(Tensor& W, Tensor& b, Node* conv, Graph& graph) {
    Value* new_W_value = graph.addInitializerAndInput(W);
    Value* old_W_value = conv->inputs()[1];
    conv->replaceInput(1, new_W_value);
    handle_old_initializer(old_W_value, graph);
    Value* new_b_value = graph.addInitializerAndInput(b);
    if (conv->inputs().size() == 3) {
      Value* old_b_value = conv->inputs()[2];
      conv->replaceInput(2, new_b_value);
      handle_old_initializer(old_b_value, graph);
    } else {
      conv->addInput(new_b_value);
    }
  }

  bool modify_conv(Node* conv, Node* bn, Graph& graph)  {
    auto bn_inputs = bn->inputs();
    auto conv_inputs = conv->inputs();
    auto bn_shape = bn->sizes();
    auto conv_shape = conv->sizes();
    auto s_index = graph.get_initializer_index(bn_inputs[1]);
    auto bbn_index = graph.get_initializer_index(bn_inputs[2]);
    auto m_index = graph.get_initializer_index(bn_inputs[3]);
    auto var_index = graph.get_initializer_index(bn_inputs[4]);
    auto W_index = graph.get_initializer_index(conv_inputs[1]);
    if (s_index == -1 || bbn_index == -1 || m_index == -1 || var_index == -1 || W_index == -1) {
      return false;
    }
    auto s = graph.get(s_index);
    auto bbn = graph.get(bbn_index);
    auto m = graph.get(m_index);
    auto var = graph.get(var_index);
    auto W = graph.get(W_index);
    auto epsilon = bn->f(kepsilon);


    Tensor* bc;
    if (conv_inputs.size() == 3) {
      auto bc_index = get_initializer_index(conv_inputs[1]);
      if (bc_index == -1) {
        return false;
      }
    }






    switch(s.elemType()) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
        //TODO: Set eps and  bc values
        Tensor eps;
        Tensor bc;

        var.apply_binary_function(add_nums<float>, eps);
        var.apply_unary_function(sqrt_num<float>)
        s.apply_binary_function(divide_nums<float>, var);
        // TODO: MAKE SURE s is raw_data
        W.scale_by_channel(s);
        bc.apply_binary_function(sub_nums<float>, m);
        bc.apply_binary_function(mult_nums<float>, s);
        bc.apply_binary_function(add_nums<float>)


        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16: {


        break;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {

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
      if (n->kind() == kBatchNormalization && n->input()->node()->kind() == kConv) {
        auto origInput = n->input();
        if (origInput->uses().size() > 1 ||
            n->outputs().size() > 1 ||
            !modify_conv(n->input()->node(), n, g)) {
          continue;
        }
        for (int i = 1; i <=4; i++)  {
          handle_old_initializer(n->inputs()[i], graph);
        }
        n->output().replaceAllUsesWith(origInput);
        it.destroyCurrent();
      }
    }
  }

  void optimize(Graph& graph) override {
    fuse_bn_into_conv(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
