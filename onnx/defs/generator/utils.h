// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

void ConstantOpInference(InferenceContext& ctx);

namespace range_detail {

class FixedUInt final {
  static constexpr size_t kWordCount = 35;
  std::array<uint64_t, kWordCount> words_{};

 public:
  static FixedUInt FromMantissa(uint64_t mantissa, size_t shift) {
    FixedUInt result;
    if (mantissa == 0) {
      return result;
    }
    const size_t word = shift / 64;
    const size_t offset = shift % 64;
    result.words_[word] = mantissa << offset;
    if (offset != 0 && word + 1 < kWordCount) {
      result.words_[word + 1] = mantissa >> (64 - offset);
    }
    return result;
  }

  int Compare(const FixedUInt& other) const {
    for (size_t i = kWordCount; i-- > 0;) {
      if (words_[i] != other.words_[i]) {
        return words_[i] < other.words_[i] ? -1 : 1;
      }
    }
    return 0;
  }

  void Add(const FixedUInt& other) {
    uint64_t carry = 0;
    for (size_t i = 0; i < kWordCount; ++i) {
      const uint64_t sum = words_[i] + other.words_[i];
      const uint64_t next = sum + carry;
      carry = static_cast<uint64_t>(sum < words_[i] || next < sum);
      words_[i] = next;
    }
  }

  void Subtract(const FixedUInt& other) {
    uint64_t borrow = 0;
    for (size_t i = 0; i < kWordCount; ++i) {
      const uint64_t current = words_[i];
      words_[i] = current - other.words_[i] - borrow;
      borrow = static_cast<uint64_t>(current < other.words_[i] || (borrow != 0 && current == other.words_[i]));
    }
  }

  FixedUInt Shifted(size_t shift) const {
    FixedUInt result;
    const size_t word_shift = shift / 64;
    const size_t bit_shift = shift % 64;
    for (size_t i = 0; i + word_shift < kWordCount; ++i) {
      result.words_[i + word_shift] |= words_[i] << bit_shift;
      if (bit_shift != 0 && i + word_shift + 1 < kWordCount) {
        result.words_[i + word_shift + 1] |= words_[i] >> (64 - bit_shift);
      }
    }
    return result;
  }

  bool IsZero() const {
    return std::all_of(words_.begin(), words_.end(), [](uint64_t word) { return word == 0; });
  }
};

template <typename T>
struct BinaryValue {
  uint64_t mantissa;
  int exponent;
  bool negative;
};

template <typename T>
BinaryValue<T> DecodeBinary(T value) {
  static_assert(std::numeric_limits<T>::radix == 2);
  static_assert(std::numeric_limits<T>::digits <= 64);
  if (value == 0) {
    return {0, 0, false};
  }
  int exponent = 0;
  const T fraction = std::frexp(std::fabs(value), &exponent);
  return {
      static_cast<uint64_t>(std::ldexp(fraction, std::numeric_limits<T>::digits)),
      exponent - std::numeric_limits<T>::digits,
      std::signbit(value)};
}

template <typename T>
int64_t ComputeFloatingRangeSize(T start, T limit, T delta) {
  const auto start_value = DecodeBinary(start);
  const auto limit_value = DecodeBinary(limit);
  const auto delta_value = DecodeBinary(delta);
  int base_exponent = delta_value.exponent;
  if (start_value.mantissa != 0) {
    base_exponent = std::min(base_exponent, start_value.exponent);
  }
  if (limit_value.mantissa != 0) {
    base_exponent = std::min(base_exponent, limit_value.exponent);
  }

  auto start_magnitude =
      FixedUInt::FromMantissa(start_value.mantissa, static_cast<size_t>(start_value.exponent - base_exponent));
  auto limit_magnitude =
      FixedUInt::FromMantissa(limit_value.mantissa, static_cast<size_t>(limit_value.exponent - base_exponent));
  const auto step =
      FixedUInt::FromMantissa(delta_value.mantissa, static_cast<size_t>(delta_value.exponent - base_exponent));

  FixedUInt distance;
  if (start_value.negative == limit_value.negative) {
    if (start_magnitude.Compare(limit_magnitude) >= 0) {
      start_magnitude.Subtract(limit_magnitude);
      distance = start_magnitude;
    } else {
      limit_magnitude.Subtract(start_magnitude);
      distance = limit_magnitude;
    }
  } else {
    start_magnitude.Add(limit_magnitude);
    distance = start_magnitude;
  }

  if (distance.Compare(step.Shifted(63)) >= 0) {
    fail_shape_inference("'Range' output size exceeds int64 limits");
  }

  uint64_t count = 0;
  for (int bit = 62; bit >= 0; --bit) {
    const auto shifted_step = step.Shifted(static_cast<size_t>(bit));
    if (distance.Compare(shifted_step) >= 0) {
      distance.Subtract(shifted_step);
      count |= uint64_t{1} << bit;
    }
  }
  if (!distance.IsZero()) {
    if (count == static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      fail_shape_inference("'Range' output size exceeds int64 limits");
    }
    ++count;
  }
  return static_cast<int64_t>(count);
}

} // namespace range_detail

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

  const T start_value = start_data[0];
  const T limit_value = limit_data[0];
  const T delta_value = delta_data[0];
  if (delta_value == 0 || !std::isfinite(start_value) || !std::isfinite(limit_value) || !std::isfinite(delta_value)) {
    fail_shape_inference("Inputs to 'Range' must be finite and delta must be non-zero");
  }

  const bool increasing = delta_value > 0;
  if ((increasing && start_value >= limit_value) || (!increasing && start_value <= limit_value)) {
    return 0;
  }
  return range_detail::ComputeFloatingRangeSize(start_value, limit_value, delta_value);
}

} // namespace ONNX_NAMESPACE
