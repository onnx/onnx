// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <limits>

#include "gtest/gtest.h"
#include "onnx/common/safe_math.h"

namespace ONNX_NAMESPACE {
namespace Test {

// ---------------------------------------------------------------------------
// checked_add_overflow
// ---------------------------------------------------------------------------

TEST(SafeMathTest, AddNoOverflow) {
  int64_t result = 0;
  EXPECT_FALSE(checked_add_overflow(3, 4, &result));
  EXPECT_EQ(result, 7);
}

TEST(SafeMathTest, AddNoOverflowNegative) {
  int64_t result = 0;
  EXPECT_FALSE(checked_add_overflow(-5, -3, &result));
  EXPECT_EQ(result, -8);
}

TEST(SafeMathTest, AddNoOverflowMixedSign) {
  int64_t result = 0;
  EXPECT_FALSE(checked_add_overflow(INT64_MAX, -1, &result));
  EXPECT_EQ(result, INT64_MAX - 1);
}

TEST(SafeMathTest, AddOverflowPositive) {
  int64_t result = 0;
  EXPECT_TRUE(checked_add_overflow(std::numeric_limits<int64_t>::max(), 1, &result));
}

TEST(SafeMathTest, AddOverflowNegative) {
  int64_t result = 0;
  EXPECT_TRUE(checked_add_overflow(std::numeric_limits<int64_t>::min(), -1, &result));
}

TEST(SafeMathTest, AddInt64MinPlusZero) {
  int64_t result = 0;
  EXPECT_FALSE(checked_add_overflow(std::numeric_limits<int64_t>::min(), 0, &result));
  EXPECT_EQ(result, std::numeric_limits<int64_t>::min());
}

// ---------------------------------------------------------------------------
// checked_sub_overflow
// ---------------------------------------------------------------------------

TEST(SafeMathTest, SubNoOverflow) {
  int64_t result = 0;
  EXPECT_FALSE(checked_sub_overflow(10, 3, &result));
  EXPECT_EQ(result, 7);
}

TEST(SafeMathTest, SubNoOverflowNegativeResult) {
  int64_t result = 0;
  EXPECT_FALSE(checked_sub_overflow(-5, 3, &result));
  EXPECT_EQ(result, -8);
}

TEST(SafeMathTest, SubNoOverflowBothNegative) {
  int64_t result = 0;
  EXPECT_FALSE(checked_sub_overflow(-3, -5, &result));
  EXPECT_EQ(result, 2);
}

TEST(SafeMathTest, SubOverflowPositiveMinusNegative) {
  // INT64_MAX - (-1) = INT64_MAX + 1 -> overflow
  int64_t result = 0;
  EXPECT_TRUE(checked_sub_overflow(std::numeric_limits<int64_t>::max(), -1, &result));
}

TEST(SafeMathTest, SubOverflowNegativeMinusPositive) {
  // INT64_MIN - 1 -> overflow
  int64_t result = 0;
  EXPECT_TRUE(checked_sub_overflow(std::numeric_limits<int64_t>::min(), 1, &result));
}

TEST(SafeMathTest, SubInt64MinMinusZero) {
  int64_t result = 0;
  EXPECT_FALSE(checked_sub_overflow(std::numeric_limits<int64_t>::min(), 0, &result));
  EXPECT_EQ(result, std::numeric_limits<int64_t>::min());
}

// ---------------------------------------------------------------------------
// checked_mul_overflow
// ---------------------------------------------------------------------------

TEST(SafeMathTest, MulNoOverflow) {
  int64_t result = 0;
  EXPECT_FALSE(checked_mul_overflow(6, 7, &result));
  EXPECT_EQ(result, 42);
}

TEST(SafeMathTest, MulNoOverflowNegative) {
  int64_t result = 0;
  EXPECT_FALSE(checked_mul_overflow(-6, 7, &result));
  EXPECT_EQ(result, -42);
}

TEST(SafeMathTest, MulNoOverflowBothNegative) {
  int64_t result = 0;
  EXPECT_FALSE(checked_mul_overflow(-6, -7, &result));
  EXPECT_EQ(result, 42);
}

TEST(SafeMathTest, MulByZero) {
  int64_t result = 99;
  EXPECT_FALSE(checked_mul_overflow(std::numeric_limits<int64_t>::max(), 0, &result));
  EXPECT_EQ(result, 0);
}

TEST(SafeMathTest, MulByOne) {
  int64_t result = 0;
  EXPECT_FALSE(checked_mul_overflow(std::numeric_limits<int64_t>::min(), 1, &result));
  EXPECT_EQ(result, std::numeric_limits<int64_t>::min());
}

TEST(SafeMathTest, MulOverflowPositive) {
  int64_t result = 0;
  EXPECT_TRUE(checked_mul_overflow(std::numeric_limits<int64_t>::max(), 2, &result));
}

TEST(SafeMathTest, MulOverflowNegativeTimesNegative) {
  // INT64_MIN * -1 = INT64_MAX + 1 -> overflow
  int64_t result = 0;
  EXPECT_TRUE(checked_mul_overflow(std::numeric_limits<int64_t>::min(), -1, &result));
}

TEST(SafeMathTest, MulInt64MinByMinusOne) {
  // Same as above, verify both argument orders
  int64_t result = 0;
  EXPECT_TRUE(checked_mul_overflow(-1, std::numeric_limits<int64_t>::min(), &result));
}

TEST(SafeMathTest, MulInt64MinByTwo) {
  int64_t result = 0;
  EXPECT_TRUE(checked_mul_overflow(std::numeric_limits<int64_t>::min(), 2, &result));
}

} // namespace Test
} // namespace ONNX_NAMESPACE
