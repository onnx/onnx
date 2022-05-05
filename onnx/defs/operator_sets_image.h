/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

// Forward declarations for ai.onnx.image version 1
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxImage, 1, CenterCropPad);

// Iterate over schema from ai.onnx.image version 1
class OpSet_OnnxImage_ver1 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxImage, 1, CenterCropPad)>());
  }
};

// Register image operators.
inline void RegisterOnnxImageOperatorSetSchema() {
  RegisterOpSetSchema<OpSet_OnnxImage_ver1>();
}

} // namespace ONNX_NAMESPACE
