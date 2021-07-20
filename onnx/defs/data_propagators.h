/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

inline void PropagateShapeDataFromInputToOutput(DataPropagationContext& ctx, int idx) {
  // propogate input data
  const auto input_data = ctx.getInputData(idx);
  if (input_data != nullptr) {
    TensorShapeProto tsp;
    tsp.CopyFrom(*input_data);
    ctx.addOutputData(0, std::move(tsp));
  }
}

}