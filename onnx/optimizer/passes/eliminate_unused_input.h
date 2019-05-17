// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   A, B, C are in the initializer list
//   D = Add(B, C)
// After:
//   B, C are in the initializer list and A is removed
//   D = Add(B, C)
//
// this pass can handle the case satisfy all following conditions:
//   condition 1: A is not used as any node's input
//   condition 2: A is not an output

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct EliminateUnusedInputs final : public FullGraphBasedPass {
  explicit EliminateUnusedInputs()
      : FullGraphBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::Memory) {}

  std::string getPassName() const override {
    return "eliminate_unused_inputs";
  }

  PassAnalysisType getPassAnalysisType() const override {
    return PassAnalysisType::Empty;
  }

  void eliminate_unused_inputs(Graph& graph) {
    while(1) {
      auto iter = std::find_if(
          graph.inputs().begin(), graph.inputs().end(), [](Value* input) { return input->uses().size() == 0; });
      if (iter != graph.inputs().end()) {
	//std::cout << "  REMOVE " << (*iter)->uniqueName() << " uses " << (*iter)->uses().size() << std::endl;
        graph.eraseInput(std::distance(graph.inputs().begin(), iter));
      } else
	break;
    }
  }

  std::shared_ptr<PostPassAnalysis> runPass(Graph& graph) override {
    eliminate_unused_inputs(graph);
    return std::shared_ptr<PostPassAnalysis>(new PostPassAnalysis());
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
