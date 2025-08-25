// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/onnx2/cpu/stream_class.h"

#include <stdint.h>

#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace ONNX_NAMESPACE::common_helpers;

namespace ONNX_NAMESPACE {
namespace v2 {

template <typename T>
uint64_t size_field(utils::BinaryWriteStream& stream, int order, const T& field, SerializeOptions& options) {
  auto s = field.SerializeSize(stream, options);
  return stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.VarintSize(s) + s;
}

template <typename T>
uint64_t size_optional_proto_field(
    utils::BinaryWriteStream& stream,
    int order,
    const utils::OptionalField<T>& field,
    SerializeOptions& options) {
  if (field.has_value()) {
    auto s = (*field).SerializeSize(stream, options);
    return stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.VarintSize(s) + s;
  }
  return 0;
}

template <>
uint64_t size_field(utils::BinaryWriteStream& stream, int order, const utils::String& field, SerializeOptions&) {
  return stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.size_string(field);
}

template <>
uint64_t size_field(
    utils::BinaryWriteStream& stream,
    int order,
    const utils::OptionalField<uint64_t>& field,
    SerializeOptions&) {
  if (field.has_value()) {
    return stream.size_field_header(order, FIELD_VARINT) + stream.size_variant_uint64(*field);
  }
  return 0;
}

template <>
uint64_t
size_field(utils::BinaryWriteStream& stream, int order, const utils::OptionalField<int64_t>& field, SerializeOptions&) {
  if (field.has_value()) {
    return stream.size_field_header(order, FIELD_VARINT) + stream.size_int64(*field);
  }
  return 0;
}

template <>
uint64_t
size_field(utils::BinaryWriteStream& stream, int order, const utils::OptionalField<int32_t>& field, SerializeOptions&) {
  if (field.has_value()) {
    return stream.size_field_header(order, FIELD_VARINT) + stream.size_int32(*field);
  }
  return 0;
}

template <>
uint64_t
size_field(utils::BinaryWriteStream& stream, int order, const utils::OptionalField<float>& field, SerializeOptions&) {
  if (field.has_value()) {
    return stream.size_field_header(order, FIELD_FIXED32) + stream.size_float(*field);
  }
  return 0;
}

template <>
uint64_t size_field(utils::BinaryWriteStream& stream, int order, const int64_t& field, SerializeOptions&) {
  return stream.size_field_header(order, FIELD_VARINT) + stream.size_int64(field);
}

template <>
uint64_t size_field(utils::BinaryWriteStream& stream, int order, const uint64_t& field, SerializeOptions&) {
  return stream.size_field_header(order, FIELD_VARINT) + stream.size_variant_uint64(field);
}

template <>
uint64_t size_field(utils::BinaryWriteStream& stream, int order, const int32_t& field, SerializeOptions&) {
  return stream.size_field_header(order, FIELD_VARINT) + stream.size_int32(field);
}

template <>
uint64_t size_field(utils::BinaryWriteStream& stream, int order, const double& field, SerializeOptions&) {
  return stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.size_double(field);
}

template <>
uint64_t size_field(utils::BinaryWriteStream& stream, int order, const float& field, SerializeOptions&) {
  return stream.size_field_header(order, FIELD_FIXED32) + stream.size_float(field);
}

template <>
uint64_t size_field(utils::BinaryWriteStream& stream, int order, const std::vector<uint8_t>& field, SerializeOptions&) {
  return stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.VarintSize(field.size()) + field.size();
}

uint64_t size_field_limit(
    utils::BinaryWriteStream& stream,
    int order,
    const std::vector<uint8_t>& field,
    SerializeOptions& options) {
  if (!options.skip_raw_data || field.size() < static_cast<size_t>(options.raw_data_threshold)) {
    if (stream.ExternalWeights() && static_cast<int64_t>(field.size()) >= options.raw_data_threshold) {
      return 0;
    } else {
      return size_field(stream, order, field, options);
    }
  }
  return 0;
}

template <typename T>
uint64_t size_enum_field(utils::BinaryWriteStream& stream, int order, const T& field, SerializeOptions&) {
  return stream.size_field_header(order, FIELD_VARINT) + stream.VarintSize(static_cast<uint64_t>(field));
}

// repeated fields

template <typename T>
uint64_t size_repeated_field(
    utils::BinaryWriteStream& stream,
    int order,
    const utils::RepeatedField<T>& field,
    bool is_packed,
    SerializeOptions& options) {
  return size_repeated_field(stream, order, field.values(), is_packed, options);
}

template <typename T>
uint64_t size_repeated_field(
    utils::BinaryWriteStream& stream,
    int order,
    const utils::RepeatedProtoField<T>& field,
    bool is_packed,
    SerializeOptions& options) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field order ", order);
  uint64_t size = 0;
  for (size_t i = 0; i < field.size(); ++i) {
    auto s = field[i].SerializeSize(stream, options);
    size += stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.VarintSize(s) + s;
  }
  return size;
}

template <typename T>
uint64_t size_repeated_field(
    utils::BinaryWriteStream& stream,
    int order,
    const std::vector<T>& field,
    bool is_packed,
    SerializeOptions& options) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field order ", order);
  uint64_t size = 0;
  for (const auto& d : field) {
    auto s = d.SerializeSize(stream, options);
    size += stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.VarintSize(s) + s;
  }
  return size;
}

template <>
uint64_t size_repeated_field(
    utils::BinaryWriteStream& stream,
    int order,
    const std::vector<utils::String>& field,
    bool is_packed,
    SerializeOptions&) {
  EXT_ENFORCE(!is_packed, "option is_packed is not implemented for field order ", order);
  uint64_t size = 0;
  for (const auto& d : field) {
    size += stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.VarintSize(d.size()) + d.size();
  }
  return size;
}

// unpacked number

template <typename T>
uint64_t size_unpacked_number_float(utils::BinaryWriteStream& stream, int order, const T&) {
  return stream.size_field_header(order, FIELD_FIXED_SIZE) + sizeof(T);
}

template <typename T>
uint64_t size_unpacked_number_int(utils::BinaryWriteStream& stream, int order, const T& value) {
  return stream.size_field_header(order, FIELD_VARINT) + stream.size_variant_uint64(static_cast<uint64_t>(value));
}

template <typename T>
uint64_t size_unpacked_number(utils::BinaryWriteStream& stream, int order, const T& value);

#define SIZE_UNPACKED_NUMBER_FLOAT(type)                                                          \
  template <>                                                                                     \
  uint64_t size_unpacked_number(utils::BinaryWriteStream& stream, int order, const type& value) { \
    return size_unpacked_number_float(stream, order, value);                                      \
  }

SIZE_UNPACKED_NUMBER_FLOAT(float)
SIZE_UNPACKED_NUMBER_FLOAT(double)

#define SIZE_UNPACKED_NUMBER_INT(type)                                                            \
  template <>                                                                                     \
  uint64_t size_unpacked_number(utils::BinaryWriteStream& stream, int order, const type& value) { \
    return size_unpacked_number_int(stream, order, value);                                        \
  }

SIZE_UNPACKED_NUMBER_INT(uint64_t)
SIZE_UNPACKED_NUMBER_INT(int64_t)
SIZE_UNPACKED_NUMBER_INT(int32_t)

// repeated packed numbers

template <typename T>
uint64_t size_repeated_field_numerical_numbers_float(
    utils::BinaryWriteStream& stream,
    int order,
    const std::vector<T>& field,
    bool,
    SerializeOptions&) {
  uint64_t size = field.size() * sizeof(T);
  return stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.size_variant_uint64(size) + size;
}

template <typename T>
uint64_t size_repeated_field_numerical_numbers_int(
    utils::BinaryWriteStream& stream,
    int order,
    const std::vector<T>& field,
    bool,
    SerializeOptions&) {
  utils::StringWriteStream local;
  uint64_t size = 0;
  for (const T& d : field) {
    size += local.size_variant_uint64(static_cast<uint64_t>(d));
  }
  return stream.size_field_header(order, FIELD_FIXED_SIZE) + stream.size_variant_uint64(size) + size;
}

template <typename T>
uint64_t size_repeated_field_numerical_numbers(
    utils::BinaryWriteStream& stream,
    int order,
    const std::vector<T>& field,
    bool is_packed,
    SerializeOptions& options);

#define SIZE_REPEATED_FIELD_NUMERICAL_FLOAT(type)                                                 \
  template <>                                                                                     \
  uint64_t size_repeated_field_numerical_numbers(                                                 \
      utils::BinaryWriteStream& stream,                                                           \
      int order,                                                                                  \
      const std::vector<type>& field,                                                             \
      bool is_packed,                                                                             \
      SerializeOptions& options) {                                                                \
    return size_repeated_field_numerical_numbers_float(stream, order, field, is_packed, options); \
  }

SIZE_REPEATED_FIELD_NUMERICAL_FLOAT(float)
SIZE_REPEATED_FIELD_NUMERICAL_FLOAT(double)

#define SIZE_REPEATED_FIELD_NUMERICAL_INT(type)                                                 \
  template <>                                                                                   \
  uint64_t size_repeated_field_numerical_numbers(                                               \
      utils::BinaryWriteStream& stream,                                                         \
      int order,                                                                                \
      const std::vector<type>& field,                                                           \
      bool is_packed,                                                                           \
      SerializeOptions& options) {                                                              \
    return size_repeated_field_numerical_numbers_int(stream, order, field, is_packed, options); \
  }

SIZE_REPEATED_FIELD_NUMERICAL_INT(int64_t)
SIZE_REPEATED_FIELD_NUMERICAL_INT(uint64_t)
SIZE_REPEATED_FIELD_NUMERICAL_INT(int32_t)

// main function to calculate size of repeated numbers

template <typename T>
uint64_t size_repeated_field_numerical(
    utils::BinaryWriteStream& stream,
    int order,
    const std::vector<T>& field,
    bool is_packed,
    SerializeOptions& options) {
  if (is_packed) {
    return size_repeated_field_numerical_numbers(stream, order, field, is_packed, options);
  } else {
    uint64_t size = 0;
    for (const T& d : field) {
      size += size_unpacked_number(stream, order, d);
    }
    return size;
  }
}

#define SIZE_REPEATED_FIELD_IMPL(type)                                              \
  template <>                                                                       \
  uint64_t size_repeated_field(                                                     \
      utils::BinaryWriteStream& stream,                                             \
      int order,                                                                    \
      const std::vector<type>& field,                                               \
      bool is_packed,                                                               \
      SerializeOptions& options) {                                                  \
    return size_repeated_field_numerical(stream, order, field, is_packed, options); \
  }

SIZE_REPEATED_FIELD_IMPL(double)
SIZE_REPEATED_FIELD_IMPL(float)
SIZE_REPEATED_FIELD_IMPL(uint64_t)
SIZE_REPEATED_FIELD_IMPL(int64_t)
SIZE_REPEATED_FIELD_IMPL(int32_t)

} // namespace v2
} // namespace ONNX_NAMESPACE