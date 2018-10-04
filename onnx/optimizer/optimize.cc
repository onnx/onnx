// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/optimizer/optimize.h"

namespace ONNX_NAMESPACE {
namespace optimization {

Optimizer Optimizer::OptimizerSingleton;

ModelProto Optimize(
    const ModelProto& mp_in,
    const std::vector<std::string>& names) {
  return Optimizer::OptimizerSingleton.optimize(mp_in, names);
}

const std::vector<std::string> GetAvailablePasses() {
  return Optimizer::OptimizerSingleton.passes.GetAvailablePasses();
}

} // namespace optimization
} // namespace ONNX_NAMESPACE
