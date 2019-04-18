// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   P = Pad(X, Pads, [Value])
//   Z = Conv(P, Y)
// After:
//   Z = Conv(X, Y) with "pads" attribute set
//
// the pass handles the case when Pad is zero-padding the input
// (i.e. mode=constant and Value=0)

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
  bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current)
      override {
    destroy_current = NodeDestroyType::DestroyZero;

    // check if Pad is only used by Conv
    if (n->inputs()[0]->uses().size() > 1) {
      return false;
    }

    Node* conv = n;
    Node* pad = n->inputs()[0]->node();

    // first check if 'pad' node has 'pads' input initialized
    const auto pads_name = pad->inputs()[1]->uniqueName();
    const auto pads_initializer = graph.getInitializer(pads_name);
    // 'pad' node has the 'pads' input which has not been initialized -
    // can't proceed with fusing
    if (pads_initializer == graph.initializers().end())
      return false;

    // parse 'pads' data from the initialized input
    std::vector<int64_t> pads;
    if (pads_initializer->elem_type() == TensorProto::INT64 &&
        pads_initializer->is_raw_data()) {
      const auto& raw_data = pads_initializer->raw();
      const size_t num_elements = static_cast<size_t>(raw_data.size() / sizeof(int64_t));
      pads.resize(num_elements);
      const int64_t* int64_data =
          reinterpret_cast<const int64_t*>(raw_data.c_str());
      for (size_t i = 0; i < num_elements; ++i) {
        pads[i] = int64_data[i];
      }
    } else if (pads_initializer->elem_type() == TensorProto::INT64) {
      pads = pads_initializer->int64s();
    }
    // not relevant data type for this input -
    // can't proceed with fusing
    else {
      return false;
    }

    std::string pad_mode;
    if (pad->hasAttribute(kmode)) {
      pad_mode = pad->s(kmode);
    } else {
      pad_mode = "constant";
    }

    double value = 0.0;
    // check if the 'pad' node has the optional 'value' input
    if (pad->inputs().size() == 3) {
      // check if it has data initialized
      const auto value_name = pad->inputs()[2]->uniqueName();
      const auto value_initializer = graph.getInitializer(value_name);

      // 'pad' node has the 'value' input which has not been initialized -
      // can't proceed with fusing
      if (value_initializer == graph.initializers().end())
        return false;

      // parse 'value' data from the initialized input
      if (value_initializer->elem_type() == TensorProto::FLOAT &&
          value_initializer->is_raw_data()) {
        const auto& raw_data = value_initializer->raw();
        value = static_cast<double>(*(reinterpret_cast<const float*>(raw_data.c_str())));
      } 
      else if (value_initializer->elem_type() == TensorProto::DOUBLE &&
          value_initializer->is_raw_data()) {
        const auto& raw_data = value_initializer->raw();
        value = *(reinterpret_cast<const double*>(raw_data.c_str()));
      }       
      else if (value_initializer->elem_type() == TensorProto::FLOAT) {
        value = static_cast<double>(value_initializer->floats()[0]);
      } else if (value_initializer->elem_type() == TensorProto::DOUBLE) {
        value = value_initializer->doubles()[0];
      }
      // either float16 or not relevant data type for this input - no fusing
      else {
        return false;
      }
    }

    // check if Pad is used to zero-pad the input
    if (pad_mode != "constant" || value != 0.0) {
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
