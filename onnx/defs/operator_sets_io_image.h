/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

// Iterate over schema from ai.onnx.io version 1
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxIO, 1, ImageDecoder);

class OpSet_OnnxIO_ver1 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxIO, 1, ImageDecoder)>());
  }
};

inline void RegisterOnnxIOOperatorSetSchema() {
  RegisterOpSetSchema<OpSet_OnnxIO_ver1>();
}
} // namespace ONNX_NAMESPACE
