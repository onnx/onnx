// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

#include <cmath>

namespace ONNX_NAMESPACE {
  void resizeShapeInference(InferenceContext& ctx);
  
  void resizeShapeInferenceHelper(
    const TensorShapeProto& input_shape,
    const std::vector<float>& scales_data,
    TensorShapeProto* output_shape
  );
}
