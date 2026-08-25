// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include "onnx/common/platform_helpers.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {

// Returns bits per element of a tensor element type, or -1 if unknown.
ONNX_API int64_t ElementBitWidth(int32_t data_type);

template <typename T>
ONNX_API TensorProto ToTensor(const T& value);

template <typename T>
ONNX_API TensorProto ToTensor(const std::vector<T>& values);

template <typename T>
std::vector<T> ParseData(const TensorProto* tensor_proto);

// Elements that dims implies, checked against raw_data's length. With exact_fit
// the two must agree exactly; otherwise raw_data may hold trailing bytes, which
// are ignored. Kept out of the template below so the error path is emitted once,
// not per element type.
ONNX_API int64_t RawDataElementCount(const TensorProto& tensor, size_t element_size, bool exact_fit);

// Decode a tensor's raw_data as values of type T. Defined here rather than in
// the .cc so callers can instantiate it for element types ParseData does not
// cover. raw_data may be unaligned for T, so it is copied byte-wise.
template <typename T>
std::vector<T> ParseRawData(const TensorProto& tensor, bool exact_fit = false) {
  static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type");
  // num_elements <= raw_data.size(), so the cast cannot truncate.
  const auto num_elements = static_cast<size_t>(RawDataElementCount(tensor, sizeof(T), exact_fit));
  std::vector<T> values(num_elements);
  if (num_elements != 0) {
    std::memcpy(values.data(), tensor.raw_data().data(), num_elements * sizeof(T));
    // raw_data is little-endian per the ONNX spec.
    if (!is_processor_little_endian()) {
      for (T& value : values) {
        auto* start = reinterpret_cast<std::byte*>(&value);
        std::reverse(start, start + sizeof(T));
      }
    }
  }
  return values;
}

} // namespace ONNX_NAMESPACE
