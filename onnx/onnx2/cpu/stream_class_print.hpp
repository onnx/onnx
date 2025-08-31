// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */
// NOLINT(readability/braces)

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
struct name_exist_value {
  const char* name;
  bool exist;
  const T* value;
  inline name_exist_value(const char* n, bool e, const T* v) : name(n), exist(e), value(v) {}
};

template <typename T>
std::string write_as_string(utils::PrintOptions&, const T& field) {
  return MakeString(field);
}

template <>
std::string write_as_string(utils::PrintOptions&, const utils::String& field) {
  return field.as_string(true);
}

template <>
std::string write_as_string(utils::PrintOptions&, const std::vector<uint8_t>& field) {
  const char* hex_chars = "0123456789ABCDEF";
  std::stringstream result;
  for (const auto& b : field) {
    result << hex_chars[b / 16] << hex_chars[b % 16];
  }
  return result.str();
}

template <typename T>
std::string write_as_string_vector(utils::PrintOptions&, const std::vector<T>& field) {
  std::stringstream result;
  result << "[";
  for (size_t i = 0; i < field.size(); ++i) {
    result << field[i];
    if (i + 1 != field.size())
      result << ", ";
  }
  result << "]";
  return result.str();
}

template <typename T>
std::string write_as_repeated_field(utils::PrintOptions&, const utils::RepeatedField<T>& field) {
  std::stringstream result;
  result << "[";
  for (size_t i = 0; i < field.size(); ++i) {
    result << field[i];
    if (i + 1 != field.size())
      result << ", ";
  }
  result << "]";
  return result.str();
}

template <>
std::string write_as_repeated_field(utils::PrintOptions&, const utils::RepeatedField<utils::String>& field) {
  std::stringstream result;
  result << "[";
  for (size_t i = 0; i < field.size(); ++i) {
    result << field[i].as_string(true);
    if (i + 1 != field.size())
      result << ", ";
  }
  result << "]";
  return result.str();
}

template <typename T>
std::string write_as_string_optional(utils::PrintOptions& options, const std::optional<T>& field) {
  if (!field)
    return "null";
  return write_as_string(options, *field);
}

template <>
std::string write_as_string(utils::PrintOptions& options, const std::vector<float>& field) {
  return write_as_string_vector(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions& options, const std::vector<int64_t>& field) {
  return write_as_string_vector(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions& options, const std::vector<uint64_t>& field) {
  return write_as_string_vector(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions& options, const std::vector<double>& field) {
  return write_as_string_vector(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions& options, const std::vector<int32_t>& field) {
  return write_as_string_vector(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions& options, const std::optional<float>& field) {
  return write_as_string_optional(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions& options, const std::optional<int64_t>& field) {
  return write_as_string_optional(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions& options, const std::optional<uint64_t>& field) {
  return write_as_string_optional(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions& options, const std::optional<double>& field) {
  return write_as_string_optional(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions& options, const std::optional<int32_t>& field) {
  return write_as_string_optional(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions& options, const utils::RepeatedField<float>& field) {
  return write_as_repeated_field(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions& options, const utils::RepeatedField<int64_t>& field) {
  return write_as_repeated_field(options, field);
}

template <>
std::string write_as_string(utils::PrintOptions& options, const utils::RepeatedField<utils::String>& field) {
  return write_as_repeated_field(options, field);
}

template <typename... Args>
std::string write_as_string(utils::PrintOptions& options, const Args&... args) {
  std::stringstream result;
  result << "{";

  auto append_arg = [&options, &result, first = true](const auto& arg) mutable {
    if (arg.exist) {
      if (!first) {
        result << ", ";
      }
      first = false;
      result << arg.name;
      result << ": ";
      auto s = write_as_string(options, *arg.value);
      if (!s.empty()) {
        if (s[s.size() - 1] == ',') {
          s.pop_back();
        }
        result << s;
      }
    }
  };

  (append_arg(args), ...);
  result << "}";
  return result.str();
}

template <typename T>
std::vector<std::string>
write_into_vector_string(utils::PrintOptions& options, const char* field_name, const T& field) {
  std::vector<std::string> r = field.PrintToVectorString(options);
  if (r.size() <= 1) {
    return {MakeString(field_name, ": ", r.back(), ",")};
  } else {
    std::vector<std::string> rows{MakeString(field_name, ": ")};
    for (size_t i = 0; i < r.size(); ++i) {
      if (i == 0) {
        rows[0] += r[0];
      } else if (i + 1 == r.size()) {
        rows.push_back(MakeString(r[i]));
      } else {
        rows.push_back(r[i]);
      }
    }
    return rows;
  }
}

template <>
std::vector<std::string>
write_into_vector_string(utils::PrintOptions& options, const char* field_name, const utils::String& field) {
  return {MakeString(field_name, ": ", write_as_string(options, field), ",")};
}

template <>
std::vector<std::string>
write_into_vector_string(utils::PrintOptions& options, const char* field_name, const int64_t& field) {
  return {MakeString(field_name, ": ", write_as_string(options, field), ",")};
}

template <>
std::vector<std::string>
write_into_vector_string(utils::PrintOptions& options, const char* field_name, const float& field) {
  return {MakeString(field_name, ": ", write_as_string(options, field), ",")};
}

template <>
std::vector<std::string>
write_into_vector_string(utils::PrintOptions& options, const char* field_name, const uint64_t& field) {
  return {MakeString(field_name, ": ", write_as_string(options, field), ",")};
}

template <>
std::vector<std::string>
write_into_vector_string(utils::PrintOptions& options, const char* field_name, const int32_t& field) {
  return {MakeString(field_name, ": ", write_as_string(options, field), ",")};
}

template <>
std::vector<std::string>
write_into_vector_string(utils::PrintOptions& options, const char* field_name, const TensorProto::DataType& field) {
  return {MakeString(field_name, ": ", write_as_string(options, static_cast<int32_t>(field)), ",")};
}

template <>
std::vector<std::string>
write_into_vector_string(utils::PrintOptions& options, const char* field_name, const TensorProto::DataLocation& field) {
  return {MakeString(field_name, ": ", write_as_string(options, static_cast<int32_t>(field)), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(
    utils::PrintOptions& options,
    const char* field_name,
    const AttributeProto::AttributeType& field) {
  return {MakeString(field_name, ": ", write_as_string(options, static_cast<int32_t>(field)), ",")};
}

template <>
std::vector<std::string>
write_into_vector_string(utils::PrintOptions& options, const char* field_name, const std::vector<uint8_t>& field) {
  return {MakeString(field_name, ": ", write_as_string(options, field), ",")};
}

template <>
std::vector<std::string> write_into_vector_string(
    utils::PrintOptions& options,
    const char* field_name,
    const utils::RepeatedField<utils::String>& field) {
  if (field.size() < 5)
    return {MakeString(field_name, ": ", write_as_string(options, field), ",")};
  std::vector<std::string> rows{MakeString(field_name, ": [")};
  for (const auto& p : field) {
    auto r = p.as_string(true);
    rows.push_back(MakeString("  ", r, ","));
  }
  rows.push_back("],");
  return rows;
}

template <typename T>
std::vector<std::string>
write_into_vector_string_repeated(utils::PrintOptions&, const char* field_name, const utils::RepeatedField<T>& field) {
  std::vector<std::string> rows;
  if (field.size() >= 10) {
    rows.push_back(MakeString(field_name, ": ["));
    for (const auto& p : field) {
      rows.push_back(MakeString("  ", p, ","));
    }
    rows.push_back("],");
  } else {
    std::vector<std::string> r;
    for (const auto& p : field) {
      r.push_back(MakeString(p));
    }
    rows.push_back(MakeString(field_name, ": [", utils::join_string(r, ", "), "],"));
  }
  return rows;
}

template <>
std::vector<std::string> write_into_vector_string(
    utils::PrintOptions& options,
    const char* field_name,
    const utils::RepeatedField<uint64_t>& field) {
  return write_into_vector_string_repeated(options, field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(
    utils::PrintOptions& options,
    const char* field_name,
    const utils::RepeatedField<int64_t>& field) {
  return write_into_vector_string_repeated(options, field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(
    utils::PrintOptions& options,
    const char* field_name,
    const utils::RepeatedField<int32_t>& field) {
  return write_into_vector_string_repeated(options, field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(
    utils::PrintOptions& options,
    const char* field_name,
    const utils::RepeatedField<float>& field) {
  return write_into_vector_string_repeated(options, field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(
    utils::PrintOptions& options,
    const char* field_name,
    const utils::RepeatedField<double>& field) {
  return write_into_vector_string_repeated(options, field_name, field);
}

template <typename T>
std::vector<std::string> write_into_vector_string_optional(
    utils::PrintOptions& options,
    const char* field_name,
    const utils::OptionalField<T>& field) {
  if (field.has_value()) {
    return {MakeString(field_name, ": ", write_as_string(options, *field), ",")};
  } else {
    return {MakeString(field_name, ": null,")};
  }
}

template <>
std::vector<std::string> write_into_vector_string(
    utils::PrintOptions& options,
    const char* field_name,
    const utils::OptionalField<int64_t>& field) {
  return write_into_vector_string_optional(options, field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(
    utils::PrintOptions& options,
    const char* field_name,
    const utils::OptionalField<uint64_t>& field) {
  return write_into_vector_string_optional(options, field_name, field);
}

template <>
std::vector<std::string> write_into_vector_string(
    utils::PrintOptions& options,
    const char* field_name,
    const utils::OptionalField<int32_t>& field) {
  return write_into_vector_string_optional(options, field_name, field);
}

template <typename... Args>
std::vector<std::string> write_proto_into_vector_string(utils::PrintOptions& options, const Args&... args) {
  std::vector<std::string> rows{"{"};
  auto append_arg = [&options, &rows](const auto& arg) mutable {
    if (arg.exist) {
      std::vector<std::string> r = write_into_vector_string(options, arg.name, *arg.value);
      for (const auto& s : r) {
        rows.push_back("  " + s);
      }
    }
  };
  (append_arg(args), ...);
  rows.push_back("},");
  return rows;
}

} // namespace v2
} // namespace ONNX_NAMESPACE
