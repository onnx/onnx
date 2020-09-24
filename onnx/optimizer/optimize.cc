// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/optimizer/optimize.h"

namespace ONNX_NAMESPACE {
namespace optimization {

GlobalPassRegistry Optimizer::passes;

Optimizer::Optimizer(
    const std::vector<std::string>& names,
    const bool fixed_point) {
  std::cout
      << "WARNING: ONNX Optimizer has been moved to https://github.com/onnx/optimizer.\n"
      << "All further enhancements and fixes to optimizers will be done in this new repo.\n"
      << "The optimizer code in onnx/onnx repo will be removed in 1.9 release.\n"
      << std::endl;
  if (fixed_point) {
    this->pass_manager =
        std::shared_ptr<FixedPointPassManager>(new FixedPointPassManager());
  } else {
    this->pass_manager =
        std::shared_ptr<GeneralPassManager>(new GeneralPassManager());
  }
  for (const auto& name : names) {
    auto pass = passes.find(name);
    this->pass_manager->add(pass);
  }
}
Optimizer::~Optimizer() {}

ModelProto Optimize(
    const ModelProto& mp_in,
    const std::vector<std::string>& names) {
  Optimizer current_opt(names, false);
  return current_opt.optimize(mp_in);
}
ModelProto OptimizeFixed(
    const ModelProto& mp_in,
    const std::vector<std::string>& names) {
  Optimizer current_opt(names, true);
  return current_opt.optimize(mp_in);
}
const std::vector<std::string> GetAvailablePasses() {
  return Optimizer::passes.GetAvailablePasses();
}

} // namespace optimization
} // namespace ONNX_NAMESPACE
