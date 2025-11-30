// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>

#include "gtest/gtest.h"
#include "onnx/common/constants.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
namespace Test {

TEST(FunctionAPITest, GetFunctionOpWithVersion) {
  const auto* const schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 9, "");
  EXPECT_TRUE(schema);
  EXPECT_TRUE(schema->HasFunction());
  const auto* const func = schema->GetFunction();
  EXPECT_EQ(func->name(), "MeanVarianceNormalization");
}

TEST(FunctionAPITest, GetMeanVarianceNormalizationFunctionWithVersion) {
  {
    const auto* const schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 13, "");
    EXPECT_TRUE(schema);
    EXPECT_TRUE(schema->HasFunction());
    const auto* const func = schema->GetFunction();
    EXPECT_EQ(func->name(), "MeanVarianceNormalization");
  }
  {
    const auto* const schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 17, "");
    EXPECT_TRUE(schema);
    EXPECT_TRUE(schema->HasFunction());
    const auto* const func = schema->GetFunction();
    EXPECT_EQ(func->name(), "MeanVarianceNormalization");
  }
  {
    const auto* const schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 18, "");
    EXPECT_TRUE(schema);
    EXPECT_TRUE(schema->HasFunction());
    const auto* const func = schema->GetFunction();
    EXPECT_EQ(func->name(), "MeanVarianceNormalization");
  }
}

} // namespace Test
} // namespace ONNX_NAMESPACE
