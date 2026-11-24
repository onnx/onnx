// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

class ONNX_EXPERIMENTAL_STATE_OPERATOR_SET_SCHEMA_CLASS_NAME(1, AttentionWithCache);
class ONNX_EXPERIMENTAL_STATE_OPERATOR_SET_SCHEMA_CLASS_NAME(1, LinearAttentionWithState);
class ONNX_EXPERIMENTAL_STATE_OPERATOR_SET_SCHEMA_CLASS_NAME(1, CausalConvWithState);

class OpSet_OnnxExperimentalState_ver1 {
 public:
  static void ForEachSchema(const std::function<void(OpSchema&&)>& fn) {
    fn(GetOpSchema<ONNX_EXPERIMENTAL_STATE_OPERATOR_SET_SCHEMA_CLASS_NAME(1, AttentionWithCache)>());
    fn(GetOpSchema<ONNX_EXPERIMENTAL_STATE_OPERATOR_SET_SCHEMA_CLASS_NAME(1, LinearAttentionWithState)>());
    fn(GetOpSchema<ONNX_EXPERIMENTAL_STATE_OPERATOR_SET_SCHEMA_CLASS_NAME(1, CausalConvWithState)>());
  }
};

ONNX_API inline void RegisterOnnxExperimentalStateOperatorSetSchema() {
  RegisterOpSetSchema<OpSet_OnnxExperimentalState_ver1>();
}

} // namespace ONNX_NAMESPACE
