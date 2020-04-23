// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

// Declare training operators.

class ONNX_EXPERIMENT_OPERATOR_SET_SCHEMA_CLASS_NAME(Gradient);
class ONNX_EXPERIMENT_OPERATOR_SET_SCHEMA_CLASS_NAME(GraphCall);
class ONNX_EXPERIMENT_OPERATOR_SET_SCHEMA_CLASS_NAME(Momentum);
class ONNX_EXPERIMENT_OPERATOR_SET_SCHEMA_CLASS_NAME(Adagrad);
class ONNX_EXPERIMENT_OPERATOR_SET_SCHEMA_CLASS_NAME(Adam);

// Iterate over schema from ai.onnx.training version 1
class OpSet_OnnxExperiment_ver1 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_EXPERIMENT_OPERATOR_SET_SCHEMA_CLASS_NAME(Gradient)>());
    fn(GetOpSchema<ONNX_EXPERIMENT_OPERATOR_SET_SCHEMA_CLASS_NAME(GraphCall)>());
    fn(GetOpSchema<ONNX_EXPERIMENT_OPERATOR_SET_SCHEMA_CLASS_NAME(Momentum)>());
    fn(GetOpSchema<ONNX_EXPERIMENT_OPERATOR_SET_SCHEMA_CLASS_NAME(Adagrad)>());
    fn(GetOpSchema<ONNX_EXPERIMENT_OPERATOR_SET_SCHEMA_CLASS_NAME(Adam)>());
  }
};

// Register experimental operators.
inline void RegisterOnnxExperimentOperatorSetSchema() {
  // Experimental operators should have only one version.
  // If changes are needed for a specific experimental operator,
  // its spec should be modified without increasing its version.
  RegisterOpSetSchema<OpSet_OnnxExperiment_ver1>();
}

} // namespace ONNX_NAMESPACE