/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

// Declare training operators.

class ONNX_PNP_OPERATOR_SET_SCHEMA_CLASS_NAME(1, LinalgSVD);

// Iterate over schema from ai.onnx.training version 1
class OpSet_OnnxPNP_ver1 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_PNP_OPERATOR_SET_SCHEMA_CLASS_NAME(1, LinalgSVD)>());
  }
};

// Register preview operators.
inline void RegisterOnnxPNPOperatorSetSchema() {
  // Preview operators should have only one version.
  // If changes are needed for a specific preview operator,
  // its spec should be modified without increasing its version.
  RegisterOpSetSchema<OpSet_OnnxPNP_ver1>();
}

} // namespace ONNX_NAMESPACE
