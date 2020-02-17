// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

// Declare training operators.
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxTraining, 1, Gradient);
class ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxTraining, 1, GraphCall);

// Iterate over schema from ai.onnx.training version 1
class OpSet_OnnxTraining_ver1 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxTraining, 1, Gradient)>());
    fn(GetOpSchema<ONNX_OPERATOR_SET_SCHEMA_CLASS_NAME(OnnxTraining, 1, GraphCall)>());
  }
};

// Register training operators.
inline void RegisterOnnxTrainingOperatorSetSchema() {
  RegisterOpSetSchema<OpSet_OnnxTraining_ver1>();
}

} // namespace ONNX_NAMESPACE