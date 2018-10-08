#include "onnx/optimizer/pass_manager.h"

namespace ONNX_NAMESPACE {
namespace optimization {

PassManager::PassManager() {}
PassManager::~PassManager() {}

GeneralPassManager::~GeneralPassManager() {
  this->passes.clear();
}
void GeneralPassManager::add(Pass* pass) {
  this->passes.insert(pass);
}

PassManagerAnalysis* GeneralPassManager::run(Graph& graph) {
  for (Pass* pass : this->passes) {
    auto pass_analysis = pass->runPass(graph);
    delete pass_analysis;
  }
  return new EmptyPassManagerAnalysis();
}

PassManagerAnalysis* FixedPointPassManager::run(Graph& graph) {
  bool fixed_point_optimization_done;

  do {
    fixed_point_optimization_done = false;
    for (Pass* pass : this->passes) {
      PostPassAnalysis* analysis = pass->runPass(graph);
      CountBasedPassAnalysis* count_analysis =
          dynamic_cast<CountBasedPassAnalysis*>(analysis);

      while (nullptr != count_analysis &&
             count_analysis->fixedPointOptimizationNeeded()) {
        count_analysis =
            dynamic_cast<CountBasedPassAnalysis*>(pass->runPass(graph));
        fixed_point_optimization_done = true;
      }
      delete count_analysis;
    }
  } while (fixed_point_optimization_done);

  return new EmptyPassManagerAnalysis();
}

} // namespace optimization
} // namespace ONNX_NAMESPACE