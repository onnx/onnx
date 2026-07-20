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

TEST(SafeMathTest, MulProductEqualsInt64MinNotFlaggedAsOverflow) {
  // INT64_MIN / 2 * 2 == INT64_MIN exactly, which is representable.
  // A naive abs-based bounds check falsely reports this as overflow because
  // abs(INT64_MIN / 2) can appear to exceed INT64_MAX / abs(2) under truncation.
  int64_t result = 0;
  const int64_t half_min = std::numeric_limits<int64_t>::min() / 2;
  EXPECT_FALSE(checked_mul_overflow(half_min, 2, &result));
  EXPECT_EQ(result, std::numeric_limits<int64_t>::min());
  EXPECT_FALSE(checked_mul_overflow(2, half_min, &result));
  EXPECT_EQ(result, std::numeric_limits<int64_t>::min());
}

TEST(SafeMathTest, MulProductOneLessThanInt64MinMagnitudeNoOverflow) {
  int64_t result = 0;
  const int64_t half_min_plus_one = std::numeric_limits<int64_t>::min() / 2 + 1;
  EXPECT_FALSE(checked_mul_overflow(half_min_plus_one, 2, &result));
  EXPECT_EQ(result, std::numeric_limits<int64_t>::min() + 2);
}

} // namespace Test
} // namespace ONNX_NAMESPACE
