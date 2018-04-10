// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Z = Conv(X, Y)
// B = Z + A
//
// the pass can handle the following cases:
//   case 1: A is 1D tensor and A.dim[0] == Z.dim[1]
//   case 2: A is 1-element 1D tensor



#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {

struct FuseAddBiasIntoConv final : public OptimizePass {
  explicit FuseAddBiasIntoConv()
    : OptimizePass("fuse_add_bias_into_conv", API_TYPE::IR) {
  }

  // NB: when numpy style broadcasting is enabled, this function will be more useful.
  static int idx_of_conv(const ArrayRef<Value *> & values) {
    for (size_t i = 0; i < values.size(); i++)
      if (values[i]->node()->kind() == kConv)
        return i;
    return -1;
  }

  void fuse_add_bias_into_conv(Graph& graph) {
    int size_lack_count = 0;
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      if (n->kind() == kAdd) {
        int idx = idx_of_conv(n->inputs());
        // due to current broadcasting's constraint, Conv has to be the first oprand
        if (idx == 0 && n->inputs()[idx]->node()->inputs().size() == 2) {
          auto orig_conv = n->inputs()[idx];
          auto orig_bias = n->inputs()[1 - idx];
          // check if bias is Const or in graph's initializers
          if (orig_bias->node()->kind() != kConstant
              && orig_bias->node()->kind() != kParam) {
            continue;
          }
          // check if conv is only used by Add
          if (orig_conv->uses().size() > 1) {
            continue;
          }
          auto conv_shape = orig_conv->sizes();
          auto bias_shape = orig_bias->sizes();
          auto weight_shape = orig_conv->node()->inputs()[1]->sizes();
          int64_t M = -1;
          // try to get feature M from conv_shape
          if (conv_shape.size() > 1 && conv_shape[1].is_int) {
            M = conv_shape[1].dim;
          }
          // try to get feature M from weight_shape
          if (weight_shape.size() > 0 && weight_shape[0].is_int) {
            ONNX_ASSERT(M == -1 || M == weight_shape[0].dim);
            M = weight_shape[0].dim;
          }
          if (M == -1 || bias_shape.size() == 0 || !bias_shape[0].is_int) {
            // No enough information, bail out
            size_lack_count += 1;
            continue;
          }
          if (bias_shape.size() == 1) {
            ONNX_ASSERT(n->hasAttribute(kbroadcast) && n->i(kbroadcast) == static_cast<int64_t>(1));
            bool able_to_optimize = bias_shape[0].dim == 1 ||
              (bias_shape[0].dim == M && (!n->hasAttribute(kaxis) || n->i(kaxis) == 1));
            if (!able_to_optimize) {
              continue;
            }
            if (bias_shape[0].dim == 1) {
              Symbol sym = Symbol("value");
              Node* constant1 = graph.create(kConstant, 1);
              Tensor t1;
              t1.sizes().push_back(static_cast<int64_t>(1));
              t1.int64s().push_back(M);
              t1.elem_type() = TensorProto_DataType_INT64;
              constant1->t_(sym, t1);
              constant1->insertBefore(orig_conv->node());
              Node* constant2 = graph.create(kConstant, 1);
              Tensor t2;
              t2.sizes().push_back(static_cast<int64_t>(1));
              t2.int64s().push_back(0);
              t2.elem_type() = TensorProto_DataType_INT64;
              constant2->t_(sym, t2);
              constant2->insertBefore(orig_conv->node());
              Node* tile = graph.create(kTile, 1);
              tile->addInput(orig_bias);
              tile->addInput(constant1->output());
              tile->addInput(constant2->output());
              tile->insertBefore(orig_conv->node());
              orig_conv->node()->addInput(tile->output());
            } else if (bias_shape[0].dim == M &&
                (!n->hasAttribute(kaxis) || n->i(kaxis) == 1)) { // default axis is 1
              orig_conv->node()->addInput(orig_bias);
            }
            if (orig_conv->sizes().size() == 0 && n->output()->sizes().size() > 0) {
              std::vector<Dimension> conv_shape(n->output()->sizes());
              orig_conv->setSizes(conv_shape);
            }
            if (n->output()->elemType() != TensorProto_DataType_UNDEFINED) {
              orig_conv->setElemType(n->output()->elemType());
            }
            n->replaceAllUsesWith(orig_conv->node());
            it.destroyCurrent();
          } else {
            continue;
          }
        }
      }
    }
    if (size_lack_count != 0) {
      std::cout <<
                "Warning: failed to fuse Add into Conv bias due to lack of size information."
                << std::endl;
    }
  }

  void optimize(Graph& graph) override {
    fuse_add_bias_into_conv(graph);
  }
};

}} // namespace ONNX_NAMESPACE::optimization
