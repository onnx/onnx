#include "onnx/optimizer/pass_manager.h"

namespace ONNX_NAMESPACE {
namespace optimization {

PassManager::PassManager() {}
PassManager::~PassManager() {}

GeneralPassManager::~GeneralPassManager() {
  this->passes.clear();
}
void GeneralPassManager::add(std::shared_ptr<Pass> pass) {
  this->passes.push_back(std::move(pass));
}

std::shared_ptr<PassManagerAnalysis> GeneralPassManager::run(Graph& graph) {
  for (std::shared_ptr<Pass> pass : this->passes) {
    auto pass_analysis = pass->runPass(graph);
  }
  return std::shared_ptr<PassManagerAnalysis>(new EmptyPassManagerAnalysis());
}

std::shared_ptr<PassManagerAnalysis> FixedPointPassManager::run(Graph& graph) {
  bool fixed_point_optimization_done;

  do {
    fixed_point_optimization_done = false;
    for (std::shared_ptr<Pass> pass : this->passes) {
      std::shared_ptr<PostPassAnalysis> analysis = pass->runPass(graph);
      if (pass->getPassAnalysisType() == PassAnalysisType::Empty) {
        continue;
      }
      std::shared_ptr<CountBasedPassAnalysis> count_analysis =
          std::static_pointer_cast<CountBasedPassAnalysis>(analysis);

      while (count_analysis->fixedPointOptimizationNeeded()) {
        count_analysis = std::static_pointer_cast<CountBasedPassAnalysis>(
            pass->runPass(graph));
        fixed_point_optimization_done = true;
      }
    }
  } while (fixed_point_optimization_done);

  return std::shared_ptr<PassManagerAnalysis>(new EmptyPassManagerAnalysis());
}
} // namespace optimization
} // namespace ONNX_NAMESPACE
