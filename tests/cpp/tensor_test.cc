// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "onnx/common/assertions.h"
#include "onnx/common/tensor.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/defs/tensor_util.h"

namespace ONNX_NAMESPACE::Test {

constexpr int64_t kLargeDim = int64_t{1} << 62;

TEST(TensorTest, ElemNumScalar) {
  Tensor t;
  EXPECT_EQ(t.elem_num(), 1);
}

TEST(TensorTest, ElemNumZeroDim) {
  Tensor t;
  t.sizes() = {0, 3};
  EXPECT_EQ(t.elem_num(), 0);
}

TEST(TensorTest, ElemNumOverflowThrows) {
#ifndef ONNX_NO_EXCEPTIONS
  Tensor t;
  t.sizes() = {kLargeDim, kLargeDim};
  EXPECT_THROW(t.elem_num(), tensor_error);
#endif
}

TEST(TensorTest, ElemNumNegativeDimThrows) {
#ifndef ONNX_NO_EXCEPTIONS
  Tensor t;
  t.sizes() = {-1, 4};
  EXPECT_THROW(t.elem_num(), tensor_error);
#endif
}

TEST(TensorTest, SizeFromDimOverflowThrows) {
#ifndef ONNX_NO_EXCEPTIONS
  Tensor t;
  t.sizes() = {2, kLargeDim, kLargeDim};
  EXPECT_THROW(t.size_from_dim(1), tensor_error);
#endif
}

// tensor_id() backs onnxoptimizer's TensorContentDigest cache
// (onnxoptimizer/passes/tensor_content_hash.h), which relies on it never
// being shared between two Tensor objects whose content could independently
// diverge -- these tests cover the identity/uniqueness guarantee that
// invariant depends on. See tensor.h's tensor_id_ comment for the full
// rationale.
TEST(TensorTest, TensorIdDistinctAcrossDefaultConstruction) {
  Tensor a;
  Tensor b;
  EXPECT_NE(a.tensor_id(), b.tensor_id());
}

TEST(TensorTest, TensorIdDistinctAcrossCopyConstruction) {
  Tensor a;
  a.sizes() = {1, 2, 3};
  Tensor b(a);
  EXPECT_NE(a.tensor_id(), b.tensor_id());
  // Copying is a deep, independent copy: mutating one afterward must not affect the other.
  b.sizes().push_back(4);
  EXPECT_NE(a.sizes(), b.sizes());
}

TEST(TensorTest, TensorIdDistinctAcrossMoveConstruction) {
  Tensor a;
  a.sizes() = {1, 2, 3};
  const uint64_t a_id = a.tensor_id();
  Tensor b(std::move(a));
  EXPECT_NE(a_id, b.tensor_id());
  EXPECT_EQ(b.sizes(), (std::vector<int64_t>{1, 2, 3}));
}

TEST(TensorTest, TensorIdRefreshedByCopyAssignment) {
  Tensor a;
  Tensor b;
  const uint64_t b_id_before = b.tensor_id();
  b = a;
  // b's content just changed, so its id must move on even if a happens to look the same.
  EXPECT_NE(b_id_before, b.tensor_id());
  EXPECT_NE(a.tensor_id(), b.tensor_id());
}

TEST(TensorTest, TensorIdRefreshedByMoveAssignment) {
  Tensor a;
  Tensor b;
  const uint64_t a_id = a.tensor_id();
  const uint64_t b_id_before = b.tensor_id();
  b = std::move(a);
  EXPECT_NE(b_id_before, b.tensor_id());
  EXPECT_NE(a_id, b.tensor_id());
}

TEST(TensorTest, TensorIdSelfAssignmentIsANoop) {
  Tensor a;
  a.sizes() = {5};
  const uint64_t id_before = a.tensor_id();
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
  a = a;
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
  EXPECT_EQ(id_before, a.tensor_id());
  EXPECT_EQ(a.sizes(), (std::vector<int64_t>{5}));
}

TEST(TensorTest, TensorIdManyConstructionsAreAllDistinct) {
  std::vector<uint64_t> ids;
  ids.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    Tensor t;
    ids.push_back(t.tensor_id());
  }
  std::vector<uint64_t> sorted_ids = ids;
  std::sort(sorted_ids.begin(), sorted_ids.end());
  EXPECT_EQ(std::adjacent_find(sorted_ids.begin(), sorted_ids.end()), sorted_ids.end())
      << "expected all 1000 tensor_id()s to be distinct";
}

TEST(TensorTest, ParseDataThrowsOnMisalignedRawData) {
#ifndef ONNX_NO_EXCEPTIONS
  Tensor t;
  // 3 bytes is not a multiple of sizeof(int32_t), so this raw_data is malformed.
  t.set_raw_data(std::string(3, '\0'));
  EXPECT_THROW(ParseData<int32_t>(&t), std::runtime_error);
#endif
}

TEST(TensorTest, ParseDataAcceptsAlignedRawData) {
  Tensor t;
  // 8 bytes is exactly two int32_t elements.
  t.set_raw_data(std::string(8, '\0'));
#ifndef ONNX_NO_EXCEPTIONS
  std::vector<int32_t> res;
  EXPECT_NO_THROW(res = ParseData<int32_t>(&t));
  EXPECT_EQ(res.size(), 2u);
#else
  EXPECT_EQ(ParseData<int32_t>(&t).size(), 2u);
#endif
}

namespace {
TensorProto MakeRawTensor(int32_t data_type, std::initializer_list<int64_t> dims, const std::string& raw_data) {
  TensorProto t;
  t.set_name("t");
  t.set_data_type(data_type);
  for (int64_t dim : dims) {
    t.add_dims(dim);
  }
  t.set_raw_data(raw_data);
  return t;
}
} // namespace

TEST(ParseRawDataTest, DecodesLittleEndianRegardlessOfHost) {
  // 1, -2 as little-endian int32.
  const TensorProto t =
      MakeRawTensor(TensorProto_DataType_INT32, {2}, std::string("\x01\0\0\0\xfe\xff\xff\xff", 8));
  const std::vector<int32_t> values = ParseRawData<int32_t>(t);
  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(values[0], 1);
  EXPECT_EQ(values[1], -2);
}

TEST(ParseRawDataTest, ToleratesTrailingBytesByDefault) {
  // dims implies one int32, raw_data holds two; the second is ignored.
  const TensorProto t =
      MakeRawTensor(TensorProto_DataType_INT32, {1}, std::string("\x01\0\0\0\x02\0\0\0", 8));
  const std::vector<int32_t> values = ParseRawData<int32_t>(t);
  ASSERT_EQ(values.size(), 1u);
  EXPECT_EQ(values[0], 1);
}

TEST(ParseRawDataTest, ExactFitRejectsTrailingBytes) {
#ifndef ONNX_NO_EXCEPTIONS
  const TensorProto t =
      MakeRawTensor(TensorProto_DataType_INT32, {1}, std::string("\x01\0\0\0\x02\0\0\0", 8));
  EXPECT_THROW(ParseRawData<int32_t>(t, /*exact_fit=*/true), std::runtime_error);
#endif
}

TEST(ParseRawDataTest, RejectsInsufficientRawData) {
#ifndef ONNX_NO_EXCEPTIONS
  const TensorProto t = MakeRawTensor(TensorProto_DataType_INT32, {4}, std::string(8, '\0'));
  EXPECT_THROW(ParseRawData<int32_t>(t), std::runtime_error);
  EXPECT_THROW(ParseRawData<int32_t>(t, /*exact_fit=*/true), std::runtime_error);
#endif
}

TEST(ParseRawDataTest, DecodesFloatBitPatterns) {
  // 1.0f, -2.0f as little-endian IEEE-754.
  const TensorProto t =
      MakeRawTensor(TensorProto_DataType_FLOAT, {2}, std::string("\0\0\x80\x3f\0\0\0\xc0", 8));
  const std::vector<float> values = ParseRawData<float>(t);
  ASSERT_EQ(values.size(), 2u);
  EXPECT_FLOAT_EQ(values[0], 1.0F);
  EXPECT_FLOAT_EQ(values[1], -2.0F);
}

TEST(ParseRawDataTest, EmptyTensorYieldsNoElements) {
  const TensorProto t = MakeRawTensor(TensorProto_DataType_INT32, {0}, "");
  EXPECT_TRUE(ParseRawData<int32_t>(t, /*exact_fit=*/true).empty());
}

} // namespace ONNX_NAMESPACE::Test
