#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct Nop final : public FullGraphBasedPass {
  explicit Nop()
      : FullGraphBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::None) {}

  std::string getPassName() const override {
    return "nop";
  }
  PostPassAnalysis runPass(Graph& graph) {
    return PostPassAnalysis();
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE