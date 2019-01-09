// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   P = Pad(X)
//   Z = Conv(P, Y)
// After:
//   Z = Conv(X, Y) with "pads" attribute set
//
// the pass handles the case when Pad is zero-padding the input
// (i.e. mode=constant and value=0)

#include <numeric>

#include "onnx/common/assertions.h"
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
  bool runTransform(Node* n, Graph& /*graph*/, NodeDestroyType& destroy_current)
      override {
    destroy_current = NodeDestroyType::DestroyZero;

    // check if Pad is only used by Conv
    if (n->inputs()[0]->uses().size() > 1) {
        return false;
    }

    Node* conv = n;
    Node* pad = n->inputs()[0]->node();

    std::string pad_mode;
    if (pad->hasAttribute(kmode)) {
      pad_mode = pad->s(kmode);
    } else {
      pad_mode = "constant";
    }
    float value = 0.0;
    if (pad->hasAttribute(kvalue)) {
      value = static_cast<float>(pad->f(kvalue));
    }

    // check if Pad is used to zero-pad the input
    if (pad_mode != "constant" || value != 0.0) {
      return false;
    }

    std::vector<int64_t> pads = pad->is(kpads);
    int pads_size = static_cast<int>(pads.size());

    // check if padding is applied only on feature dims
    if (pads[0] != 0 || pads[1] != 0 ||
        pads[pads_size / 2] != 0 || pads[pads_size / 2 + 1] != 0) {
      return false;
    }

    // check if padding is only positive
    if (std::any_of(pads.begin(), pads.end(),
        [](int64_t local_value) { return local_value < 0; })) {
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
