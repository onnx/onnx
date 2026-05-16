// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <limits>

#include "gtest/gtest.h"
#include "onnx/common/assertions.h"
#include "onnx/common/tensor.h"

namespace ONNX_NAMESPACE {
namespace Test {

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
  constexpr int64_t kLargeDim = int64_t{1} << 62;
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
  constexpr int64_t kLargeDim = int64_t{1} << 62;
  t.sizes() = {2, kLargeDim, kLargeDim};
  EXPECT_THROW(t.size_from_dim(1), tensor_error);
}

} // namespace Test
} // namespace ONNX_NAMESPACE
