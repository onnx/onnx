// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

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
