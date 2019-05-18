// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   P = Pad(X) - opset 10 and below (or) Pad(X, Pads, [Value]) - opset 11 and
//   above Z = Conv(P, Y)
// After:
//   Z = Conv(X, Y) with "pads" attribute set
//
// the pass handles the case when Pad is zero-padding the input
// (i.e. mode=constant and Value=0)

#include <numeric>

#include "onnx/defs/tensor_util.h"
#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct FusePadIntoConv final : public PredicateBasedPass {
  explicit FusePadIntoConv()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "fuse_pad_into_conv";
  }
  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kConv && node->inputs()[0]->node()->kind() == kPad;
  }
  bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current)
      override {
    destroy_current = NodeDestroyType::DestroyZero;

    // check if Pad is only used by Conv
    if (n->inputs()[0]->uses().size() > 1) {
      return false;
    }

    Node* conv = n;
    Node* pad = n->inputs()[0]->node();

    // Process 'pads' data
    std::vector<int64_t> pads;
    if (pad->hasAttribute(kpads)) {
      // opset 10 and below
      pads = pad->is(kpads);
    } else {
      // opset 11 and above - first check if 'pad' node has 'pads' input
      // initialized
      const auto pads_name = pad->inputs()[1]->uniqueName();
      const auto pads_initializer = graph.getInitializer(pads_name);
      // 'pad' node has the 'pads' input which has not been initialized -
      // can't proceed with fusing
      if (pads_initializer == graph.initializers().end()) {
        return false;
      }

      // make sure the type of 'pads' is INT64
      if (pads_initializer->elem_type() != TensorProto::INT64) {
        return false;
      }

      // parse 'pads' data from the initialized input
      pads = ParseData<int64_t>(&*pads_initializer);
    }

    // Process 'mode'
    std::string pad_mode;
    if (pad->hasAttribute(kmode)) {
      pad_mode = pad->s(kmode);
    } else {
      pad_mode = "constant";
    }

    // cannot fuse if the pad mode is not "Constant"
    if (pad_mode != "constant") {
      return false;
    }

    // Process 'value'
    double value = 0.0;
    if (pad->hasAttribute(kvalue)) {
      // opset 10 and below
      value = static_cast<double>(pad->f(kvalue));
    } else if (pad->inputs().size() == 3) {
      // opset 11 and above - check if the 'pad' node has the optional 'value'
      // input check if it has data initialized
      const auto value_name = pad->inputs()[2]->uniqueName();
      const auto value_initializer = graph.getInitializer(value_name);

      // 'pad' node has the 'value' input which has not been initialized -
      // can't proceed with fusing
      if (value_initializer == graph.initializers().end())
        return false;

      // parse 'value' data from the initialized input
      if (value_initializer->elem_type() == TensorProto::FLOAT) {
        const auto& data = ParseData<float>(&*value_initializer);
        value = static_cast<double>(data[0]);
      } else if (value_initializer->elem_type() == TensorProto::DOUBLE) {
        const auto& data = ParseData<double>(&*value_initializer);
        value = data[0];
      } else {
        // either float16 or not relevant data type for this input - no fusing
        return false;
      }
    }

    // check if Pad is used to zero-pad the input
    if (value != 0.0) {
      return false;
    }

    int pads_size = static_cast<int>(pads.size());

    // check if padding is applied only on feature dims
    if (pads[0] != 0 || pads[1] != 0 || pads[pads_size / 2] != 0 ||
        pads[pads_size / 2 + 1] != 0) {
      return false;
    }

    // check if padding is only positive
    if (std::any_of(pads.begin(), pads.end(), [](int64_t local_value) {
          return local_value < 0;
        })) {
      return false;
    }

    int conv_pads_size = pads_size - 4;
    std::vector<int64_t> conv_pads(conv_pads_size, 0);
    // Fuse into existing padding, if available
    if (conv->hasAttribute(kpads)) {
      conv_pads = conv->is(kpads);
    }

    for (int i = 2, j = 0; i < pads_size / 2; ++i, ++j) {
      conv_pads[j] += pads[i];
      conv_pads[conv_pads_size / 2 + j] += pads[pads_size / 2 + i];
    }

    conv->is_(kpads, std::move(conv_pads));
    conv->replaceInput(0, pad->inputs()[0]);
    pad->destroy();

    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
