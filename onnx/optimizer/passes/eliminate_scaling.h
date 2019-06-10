// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateScaling final : public PredicateBasedPass {
  explicit EliminateScaling()
      : PredicateBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {
    // this was not defined in the standard onnx strings
    relu_sym_ = Symbol("Relu");
  }

  std::string getPassName() const override {
    return "eliminate_scaling";
  }

  bool patternMatchPredicate(Node* node) override {
    if(node->kind() == kMul || node->kind() == kDiv) {
      if(node->inputs()[1]->node()->kind() != kParam)
	return false;

      auto prev_node = node->inputs()[0]->node();
      return(prev_node->kind() == kConv
	     || prev_node->kind() == kConvTranspose
	     || prev_node->kind() == kGemm
	     || prev_node->kind() == relu_sym_
	     || prev_node->kind() == kAdd);
    }
    return false;
  }

  bool runTransform(Node* node, Graph& graph, NodeDestroyType& destroy_current)
      override {

    destroy_current = NodeDestroyType::DestroyZero;

    auto base_node = node->inputs()[0];
    auto orig_param = node->inputs()[1];

    // check if base_node is only used by target
    if (base_node->uses().size() > 1)
      return false;

    // check if this node already has an scaling
    if(base_node->node()->hasAttribute(Symbol("__scale")))
      return false;

    // check if scale is in graph's initializers
    if (orig_param->node()->kind() != kParam ||
	orig_param->sizes().size() > 1) {
      return false;
    }
    // the scale value should be a float scalar
    Tensor T = *graph.getInitializer(orig_param->uniqueName());
    if(T.elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) // || T.size_from_dim(0) != 1)
      return false;

    float scale = T.data<float>()[0];

    // set scale value on Convolution
    if(scale != 1.0)
      base_node->node()->f_(Symbol("__scale"), scale);

    // Don't assume that theres only one output.
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      node->outputs()[i]->replaceAllUsesWith(node->input(0));
    }

    // remove initializer
    auto name = orig_param->uniqueName();
    graph.eraseInitializer(name);
    
    destroy_current = NodeDestroyType::DestroyOne;    
    return true;
  }
 private:
    Symbol relu_sym_;
};


struct EliminateActivations final : public PredicateBasedPass {
  explicit EliminateActivations()
      : PredicateBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {
    // this was not defined in the standard onnx strings
    relu_sym_ = Symbol("Relu");
    lrelu_sym_ = Symbol("LeakyRelu");
  }

  std::string getPassName() const override {
    return "eliminate_activations";
  }

  bool patternMatchPredicate(Node* node) override {

    if(node->kind() != relu_sym_
       && node->kind() != ksigmoid
       && node->kind() != kSigmoid
       && node->kind() != ktanh
       && node->kind() != kTanh
       && node->kind() != lrelu_sym_)
      return false;

    auto prev_node = node->inputs()[0]->node();
    return(prev_node->kind() == kConv
	   || prev_node->kind() == kGemm
	   || prev_node->kind() == kMatMul
	   || prev_node->kind() == kAdd
	   || prev_node->kind() == kConvTranspose);
  }

  bool runTransform(Node* node, Graph& graph, NodeDestroyType& destroy_current)
      override {

    destroy_current = NodeDestroyType::DestroyZero;

    auto base_node = node->inputs()[0];

    // check if base_node is only used by target
    if (base_node->uses().size() > 1)
      return false;

    // check if this node already has an implied activation
    if(base_node->node()->hasAttribute(Symbol("__activation")))
      return false;
      
    if(node->hasAttribute(Symbol("__scale"))) {
      auto scale = node->f(Symbol("__scale"));
      base_node->node()->f_(Symbol("__act_scale"), scale);
    }

    if(node->kind() == relu_sym_)
      base_node->node()->s_(Symbol("__activation"), "relu");
    else if(node->kind() == lrelu_sym_)
      base_node->node()->s_(Symbol("__activation"), "leakyrelu");
    else if(node->kind() == ksigmoid || node->kind() == kSigmoid)
      base_node->node()->s_(Symbol("__activation"), "sigmoid");
    else
      base_node->node()->s_(Symbol("__activation"), "unknownactivation");

    // Don't assume that theres only one output.
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      node->outputs()[i]->replaceAllUsesWith(node->input(0));
    }

    destroy_current = NodeDestroyType::DestroyOne;    
    return true;
  }
 private:
    Symbol relu_sym_;
    Symbol lrelu_sym_;
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
