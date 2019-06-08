// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {
  void resizeShapeInference(InferenceContext& ctx);
  void upsampleShapeInferenceV7(InferenceContext& ctx);
}