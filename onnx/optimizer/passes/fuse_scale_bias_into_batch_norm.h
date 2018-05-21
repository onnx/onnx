// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// TODO
// Before:
//   Z = BatchNorm(X, 1, 0, mean, var)
//   S = Mul(Z, scale)
//   B = Add(S, bias)
// After:
//   B = BatchNorm(X, scale, bias, mean, var)
//
// the pass can handle the following cases:
// (scale and bias should both match cases)
//   for scale:
//   case 1: scale is 1D tensor and scale.dim[0] == Z.dim[1]
//   case 2: scale is 1-element 1D tensor
//   for bias:
//   case 1: bias is 1D tensor and bias.dim[0] == S.dim[1]
//   case 2: bias is 1-element 1D tensor

#include <numeric>

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FuseScaleBiasIntoBatchNorm final : public OptimizePass {
  explicit FuseScaleBiasIntoBatchNorm()
      : OptimizePass("fuse_scale_bias_into_batch_norm", API_TYPE::IR) {}

  void insert_squeeze(
      Graph& graph,
      std::vector<Dimension>& shape,
      Value* input,
      Value* target) {
    Node* squeeze = graph.create(kSqueeze, 1);
    std::vector<int64_t> axes(shape.size() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    squeeze->is_(kaxes, std::move(axes));
    squeeze->addInput(input);
    input = squeeze->output();
    squeeze->insertBefore(target->node());
  }

  void insert_constant_tile(Graph& graph, Value* input, Value* target, int64_t M) {
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
    constant->insertBefore(target->node());
    Node* tile = graph.create(kTile, 1);
    tile->addInput(input);
    tile->addInput(constant->output());
    input = tile->output();
    tile->insertBefore(target->node());
  }

  void fuse_scale_bias_into_batch_norm(Graph& graph) {
    int size_lack_count = 0;
    for (auto it = graph.begin(); it != graph.end(); ++it) {
      auto* n = *it;
      DescendOnGraphAttributes(
          n, [this](Graph& g) { fuse_scale_bias_into_batch_norm(g); });
      if (n->kind() == kAdd && n->inputs()[0]->node()->kind() == kMul &&
          n->inputs()[0]->node()->inputs()[0]->node()->kind() ==
              kBatchNormalization) {
        auto orig_bias = n->inputs()[1];
        auto orig_scale = n->inputs()[0]->node()->inputs()[1];
        auto orig_batch_norm = n->inputs()[0]->node()->inputs()[0];
        // check if bias is Const or in graph's initializers
        if (orig_bias->node()->kind() != kConstant &&
            orig_bias->node()->kind() != kParam) {
          continue;
        }
        // check if scale is Const or in graph's initializers
        if (orig_scale->node()->kind() != kConstant &&
            orig_scale->node()->kind() != kParam) {
          continue;
        }
        // check if BatchNorm is only used by Mul
        if (orig_batch_norm->uses().size() > 1) {
          continue;
        }
        // check if Mul is only used by Add
        if (n->inputs()[0]->uses().size() > 1) {
          continue;
        }
        auto batch_norm_shape = orig_batch_norm->sizes();
        auto scale_shape = orig_scale->sizes();
        auto bias_shape = orig_bias->sizes();

        int64_t M = -1;
        int64_t rank = -1;
        // try to get feature M and rank from batch_norm_shape
        if (batch_norm_shape.size() > 1 && batch_norm_shape[1].is_int) {
          M = batch_norm_shape[1].dim;
          rank = batch_norm_shape.size();
        }

        int64_t num_el_scale = 1;
        for (int i = 0; i < static_cast<int64_t>(scale_shape.size()); ++i) {
          if (scale_shape[i].is_int) {
            num_el_scale *= scale_shape[i].dim;
          } else {
            num_el_scale = -1;
            break;
          }
        }

        int64_t num_el_bias = 1;
        for (int i = 0; i < static_cast<int64_t>(bias_shape.size()); ++i) {
          if (bias_shape[i].is_int) {
            num_el_bias *= bias_shape[i].dim;
          } else {
            num_el_bias = -1;
            break;
          }
        }
        if (M == -1 || num_el_scale == -1 || num_el_bias == -1) {
          // No enough information, bail out
          size_lack_count += 1;
          continue;
        }

        if (rank < static_cast<int64_t>(scale_shape.size()) ||
            rank < static_cast<int64_t>(bias_shape.size())) {
          continue;
        }

        if (num_el_scale == 1) {
        } else if (rank > static_cast<int64_t>(scale_shape.size()) + 1) {
          continue;
        } else if (
            num_el_scale == M &&
            scale_shape[1 + scale_shape.size() - static_cast<unsigned>(rank)]
                    .dim == M) {
        } else {
          continue;
        }

        if (num_el_bias == 1) {
        } else if (rank > static_cast<int64_t>(bias_shape.size()) + 1) {
          continue;
        } else if (
            num_el_bias == M &&
            bias_shape[1 + bias_shape.size() - static_cast<unsigned>(rank)]
                    .dim == M) {
        } else {
          continue;
        }

        if (orig_scale->node()->kind() != kParam &&
            orig_batch_norm->node()->isBefore(orig_scale->node())) {
          orig_scale->node()->moveBefore(orig_batch_norm->node());
        }
        if (num_el_scale == 1) {
          Value* batch_norm_2nd_input = orig_scale;
          if (scale_shape.size() > 1) {
            insert_squeeze(
                graph, scale_shape, batch_norm_2nd_input, orig_batch_norm);
          }
          if (M > 1) {
            insert_constant_tile(graph, batch_norm_2nd_input, orig_batch_norm, M);
          }
          orig_batch_norm->node()->replaceInputWith(
              orig_batch_norm->node()->inputs()[1], batch_norm_2nd_input);
        } else if (rank > static_cast<int64_t>(scale_shape.size()) + 1) {
          continue;
        } else if (
            num_el_scale == M &&
            scale_shape[1 + scale_shape.size() - static_cast<unsigned>(rank)]
                    .dim == M) {
          ONNX_ASSERT(scale_shape.size() > 1);
          if (orig_scale->node()->kind() != kParam &&
              orig_batch_norm->node()->isBefore(orig_scale->node())) {
            orig_scale->node()->moveBefore(orig_batch_norm->node());
          }
          Node* squeeze = graph.create(kSqueeze, 1);
          std::vector<int64_t> axes(scale_shape.size());
          std::iota(axes.begin(), axes.end(), static_cast<int64_t>(0));
          axes.erase(
              axes.begin() + 1 + scale_shape.size() -
              static_cast<unsigned>(rank));
          squeeze->is_(kaxes, std::move(axes));
          squeeze->addInput(orig_scale);
          squeeze->insertBefore(orig_batch_norm->node());
          orig_batch_norm->node()->replaceInputWith(
              orig_batch_norm->node()->inputs()[1], squeeze->output());
        } else {
          continue;
        }

        if (orig_bias->node()->kind() != kParam &&
            orig_batch_norm->node()->isBefore(orig_bias->node())) {
          orig_bias->node()->moveBefore(orig_batch_norm->node());
        }
        if (num_el_bias == 1) {
          Value* batch_norm_3rd_input = orig_bias;
          if (bias_shape.size() > 1) {
            insert_squeeze(
                graph, bias_shape, batch_norm_3rd_input, orig_batch_norm);
          }
          if (M > 1) {
            insert_constant_tile(graph, batch_norm_3rd_input, orig_batch_norm, M);
          }
          orig_batch_norm->node()->replaceInputWith(
              orig_batch_norm->node()->inputs()[2], batch_norm_3rd_input);
        } else if (rank > static_cast<int64_t>(bias_shape.size()) + 1) {
          continue;
        } else if (
            num_el_bias == M &&
            bias_shape[1 + bias_shape.size() - static_cast<unsigned>(rank)]
                    .dim == M) {
          ONNX_ASSERT(bias_shape.size() > 1);
          Node* squeeze = graph.create(kSqueeze, 1);
          std::vector<int64_t> axes(bias_shape.size());
          std::iota(axes.begin(), axes.end(), static_cast<int64_t>(0));
          axes.erase(
              axes.begin() + 1 + bias_shape.size() -
              static_cast<unsigned>(rank));
          squeeze->is_(kaxes, std::move(axes));
          squeeze->addInput(orig_bias);
          squeeze->insertBefore(orig_batch_norm->node());
          orig_batch_norm->node()->replaceInputWith(
              orig_batch_norm->node()->inputs()[2], squeeze->output());
        } else {
          continue;
        }

        if (orig_batch_norm->sizes().size() == 0 &&
            n->output()->sizes().size() > 0) {
          orig_batch_norm->setSizes(n->output()->sizes());
        }
        if (n->output()->elemType() != TensorProto_DataType_UNDEFINED) {
          orig_batch_norm->setElemType(n->output()->elemType());
        }
        n->replaceAllUsesWith(orig_batch_norm->node());
        n->inputs()[0]->node()->destroy();
        it.destroyCurrent();
      }
    }
    if (size_lack_count != 0) {
      std::cout
          << "Warning: failed to fuse Mul and Add into BatchNorm scale and bias due to lack of size information."
          << std::endl;
    }
  }

  void optimize(Graph& graph) override {
    fuse_scale_bias_into_batch_norm(graph);
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE