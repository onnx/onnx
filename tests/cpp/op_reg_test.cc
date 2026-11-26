// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "onnx/defs/data_type_utils.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE::Test {
TEST(OpRegistrationTest, GemmOp) {
  const auto* const opSchema = OpSchemaRegistry::Schema("Gemm");
  ASSERT_TRUE(nullptr != opSchema);
  size_t input_size = opSchema->inputs().size();
  EXPECT_EQ(input_size, 3);
  EXPECT_EQ(opSchema->inputs()[0].GetTypes(), opSchema->outputs()[0].GetTypes());
  size_t attr_size = opSchema->attributes().size();
  EXPECT_EQ(attr_size, 4);
  EXPECT_NE(opSchema->attributes().count("alpha"), 0);
  EXPECT_EQ(opSchema->attributes().at("alpha").type, AttributeProto_AttributeType_FLOAT);
  EXPECT_NE(opSchema->attributes().count("beta"), 0);
  EXPECT_EQ(opSchema->attributes().at("beta").type, AttributeProto_AttributeType_FLOAT);
  for (const auto& parameter : opSchema->typeConstraintParams()) {
    const auto& constraint = opSchema->typeConstraintMap().at(parameter.type_param_str);
    EXPECT_EQ(constraint.second, parameter.description);
    EXPECT_EQ(constraint.first.size(), parameter.allowed_type_strs.size());
    for (const auto& type : parameter.allowed_type_strs) {
      EXPECT_EQ(constraint.first.count(Utils::DataTypeUtils::ToType(type)), 1u);
    }
  }
}
} // namespace ONNX_NAMESPACE::Test
