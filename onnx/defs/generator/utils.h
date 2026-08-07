// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

void ConstantOpInference(InferenceContext& ctx);

template <typename T>
int64_t compute_output_dim_for_range(const TensorProto* start, const TensorProto* limit, const TensorProto* delta) {
  if (!start->dims().empty() || !limit->dims().empty() || !delta->dims().empty()) {
    fail_shape_inference("Input to 'Range' op should be scalars (Tensor with only one element and shape empty)");
  }

  const auto start_data = ParseData<T>(start);
  const auto limit_data = ParseData<T>(limit);
  const auto delta_data = ParseData<T>(delta);

  const long double start_value = static_cast<long double>(start_data[0]);
  const long double limit_value = static_cast<long double>(limit_data[0]);
  const long double delta_value = static_cast<long double>(delta_data[0]);
  if (delta_value == 0 || !std::isfinite(start_value) || !std::isfinite(limit_value) ||
      !std::isfinite(delta_value)) {
    fail_shape_inference("Inputs to 'Range' must be finite and delta must be non-zero");
  }

  const long double count = std::ceil((limit_value - start_value) / delta_value);
  if (count <= 0) {
    return 0;
  }
  if (count > static_cast<long double>(std::numeric_limits<int64_t>::max())) {
    fail_shape_inference("'Range' output size exceeds int64 limits");
  }
  return static_cast<int64_t>(count);
}

} // namespace ONNX_NAMESPACE
