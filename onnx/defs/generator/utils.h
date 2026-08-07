// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

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

  if constexpr (std::is_integral_v<T>) {
    const T start_value = start_data[0];
    const T limit_value = limit_data[0];
    const T delta_value = delta_data[0];
    if (delta_value == 0) {
      fail_shape_inference("Input 'delta' to 'Range' must be non-zero");
    }

    const bool increasing = delta_value > 0;
    if ((increasing && start_value >= limit_value) || (!increasing && start_value <= limit_value)) {
      return 0;
    }

    const uint64_t distance = increasing ? static_cast<uint64_t>(limit_value) - static_cast<uint64_t>(start_value)
                                         : static_cast<uint64_t>(start_value) - static_cast<uint64_t>(limit_value);
    const uint64_t step =
        increasing ? static_cast<uint64_t>(delta_value) : uint64_t{0} - static_cast<uint64_t>(delta_value);
    const uint64_t count = distance / step + static_cast<uint64_t>(distance % step != 0);
    if (count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      fail_shape_inference("'Range' output size exceeds int64 limits");
    }
    return static_cast<int64_t>(count);
  }

  const long double start_value = static_cast<long double>(start_data[0]);
  const long double limit_value = static_cast<long double>(limit_data[0]);
  const long double delta_value = static_cast<long double>(delta_data[0]);
  if (delta_value == 0 || !std::isfinite(start_value) || !std::isfinite(limit_value) || !std::isfinite(delta_value)) {
    fail_shape_inference("Inputs to 'Range' must be finite and delta must be non-zero");
  }

  const long double count = std::ceil((limit_value - start_value) / delta_value);
  if (count <= 0) {
    return 0;
  }
  constexpr long double kInt64ExclusiveUpperBound = 9223372036854775808.0L;
  if (count >= kInt64ExclusiveUpperBound) {
    fail_shape_inference("'Range' output size exceeds int64 limits");
  }
  return static_cast<int64_t>(count);
}

} // namespace ONNX_NAMESPACE
