#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct NopEmptyPass final : public FullGraphBasedPass {
  explicit NopEmptyPass()
      : FullGraphBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::None) {}

  std::string getPassName() const override {
    return "nop";
  }
  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::Empty;
  }
  std::shared_ptr<PostPassAnalysis> runPass(Graph&) override {
    return std::make_shared<PostPassAnalysis>();
  }
};
} // namespace optimization
} // namespace ONNX_NAMESPACE
