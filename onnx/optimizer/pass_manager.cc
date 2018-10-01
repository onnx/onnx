#include "onnx/optimizer/pass_manager.h"

namespace ONNX_NAMESPACE {
namespace optimization {

PassManager::PassManager() {}
PassManager::~PassManager() {}

void GeneralPassManager::add(Pass* pass) {
  this->passes.insert(pass);
}
PassManagerAnalysis GeneralPassManager::run(Graph& graph) {
  for (Pass* pass : this->passes) {
    pass->runPass(graph);
  }
  return EmptyPassManagerAnalysis();
}

} // namespace optimization
} // namespace ONNX_NAMESPACE