// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "onnx/defs/data_type_utils.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE::types {

// Authoring DSL for op schema type strings. Factories return canonical
// std::string forms ("tensor(float)", "seq(...)", "map(int64, ...)") that
// drop into the existing string-based OpSchema API. Output is byte-identical
// to hand-written strings.

namespace internal {
// Element-type name as it appears inside tensor(...) / sparse_tensor(...) /
// map(...). Propagates std::invalid_argument if `e` is an unknown enum.
inline std::string ElemStr(TensorProto::DataType e) {
  return Utils::DataTypeUtils::ToDataTypeString(static_cast<int32_t>(e));
}
} // namespace internal

inline std::string Tensor(TensorProto::DataType e) {
  return "tensor(" + internal::ElemStr(e) + ")";
}

inline std::string SparseTensor(TensorProto::DataType e) {
  return "sparse_tensor(" + internal::ElemStr(e) + ")";
}

inline std::string Sequence(std::string inner) {
  return "seq(" + std::move(inner) + ")";
}

inline std::string Optional(std::string inner) {
  return "optional(" + std::move(inner) + ")";
}

// `Key` must be an integer type or STRING (the ONNX-supported map key types).
// Enforced at compile time via static_assert.
template <TensorProto::DataType Key>
inline std::string Map(std::string value) {
  static_assert(
      Key == TensorProto::INT8 || Key == TensorProto::INT16 || Key == TensorProto::INT32 || Key == TensorProto::INT64 ||
          Key == TensorProto::UINT8 || Key == TensorProto::UINT16 || Key == TensorProto::UINT32 ||
          Key == TensorProto::UINT64 || Key == TensorProto::STRING,
      "invalid ONNX map key type — must be an integer type or STRING");
  // Space after the comma matches existing hand-written schema sources.
  return "map(" + internal::ElemStr(Key) + ", " + std::move(value) + ")";
}

// Prebuilt "tensor(<elem>)" strings — one per element type.
inline constexpr const char* Float16 = "tensor(float16)";
inline constexpr const char* Float = "tensor(float)";
inline constexpr const char* Double = "tensor(double)";
inline constexpr const char* BFloat16 = "tensor(bfloat16)";
inline constexpr const char* Int8 = "tensor(int8)";
inline constexpr const char* Int16 = "tensor(int16)";
inline constexpr const char* Int32 = "tensor(int32)";
inline constexpr const char* Int64 = "tensor(int64)";
inline constexpr const char* UInt8 = "tensor(uint8)";
inline constexpr const char* UInt16 = "tensor(uint16)";
inline constexpr const char* UInt32 = "tensor(uint32)";
inline constexpr const char* UInt64 = "tensor(uint64)";
inline constexpr const char* Bool = "tensor(bool)";
inline constexpr const char* String = "tensor(string)";
inline constexpr const char* Complex64 = "tensor(complex64)";
inline constexpr const char* Complex128 = "tensor(complex128)";
inline constexpr const char* Float8E4M3FN = "tensor(float8e4m3fn)";
inline constexpr const char* Float8E4M3FNUZ = "tensor(float8e4m3fnuz)";
inline constexpr const char* Float8E5M2 = "tensor(float8e5m2)";
inline constexpr const char* Float8E5M2FNUZ = "tensor(float8e5m2fnuz)";
inline constexpr const char* Float4E2M1 = "tensor(float4e2m1)";
inline constexpr const char* Float8E8M0 = "tensor(float8e8m0)";
inline constexpr const char* UInt4 = "tensor(uint4)";
inline constexpr const char* Int4 = "tensor(int4)";
inline constexpr const char* UInt2 = "tensor(uint2)";
inline constexpr const char* Int2 = "tensor(int2)";

// Vector helpers: build type-string lists from element types, or wrap each
// entry of an existing list with seq(...) / optional(...).

inline std::vector<std::string> Tensors(std::initializer_list<TensorProto::DataType> elems) {
  std::vector<std::string> out;
  out.reserve(elems.size());
  for (auto e : elems) {
    out.push_back(Tensor(e));
  }
  return out;
}

inline std::vector<std::string> SparseTensors(std::initializer_list<TensorProto::DataType> elems) {
  std::vector<std::string> out;
  out.reserve(elems.size());
  for (auto e : elems) {
    out.push_back(SparseTensor(e));
  }
  return out;
}

inline std::vector<std::string> Sequence(const std::vector<std::string>& inner) {
  std::vector<std::string> out;
  out.reserve(inner.size());
  for (const auto& s : inner) {
    out.push_back(Sequence(s));
  }
  return out;
}

inline std::vector<std::string> Optional(const std::vector<std::string>& inner) {
  std::vector<std::string> out;
  out.reserve(inner.size());
  for (const auto& s : inner) {
    out.push_back(Optional(s));
  }
  return out;
}

inline std::vector<std::string> Concat(std::vector<std::string> a, const std::vector<std::string>& b) {
  a.reserve(a.size() + b.size());
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

} // namespace ONNX_NAMESPACE::types
