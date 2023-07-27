/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

// Iterate over schema from ai.onnx.io.image version 1
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxIOImage, 1, ImageDecoder);

class OpSet_OnnxIOImage_ver1 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxIOImage, 1, ImageDecoder)>());
  }
};

inline void RegisterOnnxIOImageOperatorSetSchema() {
  RegisterOpSetSchema<OpSet_OnnxIOImage_ver1>();
}
} // namespace ONNX_NAMESPACE
