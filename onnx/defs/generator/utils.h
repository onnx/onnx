// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>

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

  int64_t n = static_cast<int64_t>(ceil((1.0 * (limit_data[0] - start_data[0])) / delta_data[0]));

  n = std::max<int64_t>(n, 0);

  return n;
}

} // namespace ONNX_NAMESPACE
