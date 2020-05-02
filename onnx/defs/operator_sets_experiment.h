// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

// Declare training operators.

class ONNX_EXPERIMENTAL_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Gradient);
class ONNX_EXPERIMENTAL_OPERATOR_SET_SCHEMA_CLASS_NAME(1, GraphCall);
class ONNX_EXPERIMENTAL_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Momentum);
class ONNX_EXPERIMENTAL_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Adagrad);
class ONNX_EXPERIMENTAL_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Adam);

// Iterate over schema from ai.onnx.training version 1
class OpSet_OnnxExperimental_ver1 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_EXPERIMENTAL_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Gradient)>());
    fn(GetOpSchema<ONNX_EXPERIMENTAL_OPERATOR_SET_SCHEMA_CLASS_NAME(1, GraphCall)>());
    fn(GetOpSchema<ONNX_EXPERIMENTAL_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Momentum)>());
    fn(GetOpSchema<ONNX_EXPERIMENTAL_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Adagrad)>());
    fn(GetOpSchema<ONNX_EXPERIMENTAL_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Adam)>());
  }
};

// Register experimental operators.
inline void RegisterOnnxExperimentalOperatorSetSchema() {
  // Experimental operators should have only one version.
  // If changes are needed for a specific experimental operator,
  // its spec should be modified without increasing its version.
  RegisterOpSetSchema<OpSet_OnnxExperimental_ver1>();
}

} // namespace ONNX_NAMESPACE