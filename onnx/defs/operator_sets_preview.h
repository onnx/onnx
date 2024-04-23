/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

// Declare training operators version 1

class ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Gradient);
class ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Momentum);
class ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Adagrad);
class ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Adam);

// Iterate over schema from ai.onnx.training version 1
class OpSet_OnnxPreview_ver1 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Gradient)>());
    fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Momentum)>());
    fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Adagrad)>());
    fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(1, Adam)>());
  }
};

// Declare training operators version 2

class ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(2, Gradient);

// Iterate over schema from ai.onnx.training version 2
class OpSet_OnnxPreview_ver2 {
 public:
  static void ForEachSchema(std::function<void(OpSchema&&)> fn) {
    fn(GetOpSchema<ONNX_PREVIEW_OPERATOR_SET_SCHEMA_CLASS_NAME(2, Gradient)>());
  }
};

// Register preview operators.
inline void RegisterOnnxPreviewOperatorSetSchema() {
  // Preview operators should have only one version.
  // If changes are needed for a specific preview operator,
  // its spec should be modified without increasing its version.
  RegisterOpSetSchema<OpSet_OnnxPreview_ver1>();
  RegisterOpSetSchema<OpSet_OnnxPreview_ver2>();
}

} // namespace ONNX_NAMESPACE
