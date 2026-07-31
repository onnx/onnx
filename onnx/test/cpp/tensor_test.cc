// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <limits>
#include <string>

#include "gtest/gtest.h"
#include "onnx/common/assertions.h"
#include "onnx/common/tensor.h"
#include "onnx/defs/tensor_util.h"

namespace ONNX_NAMESPACE {
namespace Test {

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
  Tensor t;
  t.sizes() = {kLargeDim, kLargeDim};
  EXPECT_THROW(t.elem_num(), tensor_error);
}

TEST(TensorTest, ElemNumNegativeDimThrows) {
  Tensor t;
  t.sizes() = {-1, 4};
  EXPECT_THROW(t.elem_num(), tensor_error);
}

TEST(TensorTest, SizeFromDimOverflowThrows) {
  Tensor t;
  t.sizes() = {2, kLargeDim, kLargeDim};
  EXPECT_THROW(t.size_from_dim(1), tensor_error);
}

TEST(TensorTest, ParseDataThrowsOnMisalignedRawData) {
  Tensor t;
  // 3 bytes is not a multiple of sizeof(int32_t), so this raw_data is malformed.
  t.set_raw_data(std::string(3, '\0'));
  EXPECT_THROW(ParseData<int32_t>(&t), std::runtime_error);
}

TEST(TensorTest, ParseDataAcceptsAlignedRawData) {
  Tensor t;
  // 8 bytes is exactly two int32_t elements.
  t.set_raw_data(std::string(8, '\0'));
  std::vector<int32_t> res;
  EXPECT_NO_THROW(res = ParseData<int32_t>(&t));
  EXPECT_EQ(res.size(), 2u);
}

} // namespace Test
} // namespace ONNX_NAMESPACE
