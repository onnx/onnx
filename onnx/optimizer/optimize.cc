// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/optimizer/optimize.h"

namespace ONNX_NAMESPACE { namespace optimization {

static Optimizer _optimizer;

ModelProto Optimize(
    const ModelProto& mp_in,
    const std::vector<std::string>& names) {
  return _optimizer.optimize(mp_in, names);
}

const std::vector<std::string> GetAvailablePasses() {
  std::vector<std::string> names;
  for (const auto& pass : _optimizer.passes) {
    names.push_back(pass.first);
  }
  return names;
}

}} // namespace ONNX_NAMESPACE:optimization
