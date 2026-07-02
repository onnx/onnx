// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#include "onnx/defs/tensor_util.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "onnx/common/platform_helpers.h"

namespace ONNX_NAMESPACE {

#define DEFINE_PARSE_DATA(type, typed_data_fetch)                                             \
  template <>                                                                                 \
  std::vector<type> ParseData(const Tensor* tensor) {                                         \
    std::vector<type> res;                                                                    \
    if (!tensor->is_raw_data()) {                                                             \
      const auto& data = tensor->typed_data_fetch();                                          \
      res.insert(res.end(), data.begin(), data.end());                                        \
      return res;                                                                             \
    }                                                                                         \
    const std::string& raw_data = tensor->raw();                                              \
    /* copy byte-wise: raw_data may be unaligned for type */                                  \
    /* require a whole number of elements */                                                  \
    ONNX_ASSERTM(                                                                             \
        raw_data.size() % sizeof(type) == 0,                                                  \
        "raw_data size %zu is not a multiple of element size %zu",                            \
        raw_data.size(),                                                                      \
        sizeof(type));                                                                        \
    res.resize(raw_data.size() / sizeof(type));                                               \
    std::byte* bytes = reinterpret_cast<std::byte*>(res.data());                              \
    std::copy_n(reinterpret_cast<const std::byte*>(raw_data.data()), raw_data.size(), bytes); \
    /* swap byte order on big-endian hosts */                                                 \
    if (!is_processor_little_endian()) {                                                      \
      for (auto& element : res) {                                                             \
        std::byte* start_byte = reinterpret_cast<std::byte*>(&element);                       \
        std::reverse(start_byte, start_byte + sizeof(type));                                  \
      }                                                                                       \
    }                                                                                         \
    return res;                                                                               \
  }

DEFINE_PARSE_DATA(int32_t, int32s)
DEFINE_PARSE_DATA(int64_t, int64s)
DEFINE_PARSE_DATA(float, floats)
DEFINE_PARSE_DATA(double, doubles)
DEFINE_PARSE_DATA(uint64_t, uint64s)

#undef DEFINE_PARSE_DATA

} // namespace ONNX_NAMESPACE
