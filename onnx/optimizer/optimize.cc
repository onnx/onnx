// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/optimizer/optimize.h"

namespace ONNX_NAMESPACE { namespace optimization {

static Optimizer _optimizer;

ONNX_NAMESPACE::ModelProto Optimize(
    const ONNX_NAMESPACE::ModelProto& mp_in,
    const std::vector<std::string>& names) {
  return _optimizer.optimize(mp_in, names);
}

}}
