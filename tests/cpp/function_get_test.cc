// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "gtest/gtest.h"
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE::Test {

TEST(FunctionAPITest, GetFunctionOpWithVersion) {
  const auto* const schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 9, "");
  EXPECT_TRUE(schema);
  EXPECT_TRUE(schema->HasFunction());
  const auto* const func = schema->GetFunction();
  EXPECT_EQ(func->name(), "MeanVarianceNormalization");
}

TEST(FunctionAPITest, GetMeanVarianceNormalizationFunctionWithVersion) {
  for (const int version : {13, 17, 18, 28}) {
    const auto* const schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", version, "");
    ASSERT_TRUE(schema);
    EXPECT_TRUE(schema->HasFunction());
    EXPECT_FALSE(schema->HasContextDependentFunction());
    EXPECT_EQ(schema->GetFunction()->name(), "MeanVarianceNormalization");
  }

  NodeProto node;
  node.set_op_type("MeanVarianceNormalization");
  node.add_input("X");
  node.add_output("Y");
  TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  FunctionBodyBuildContextImpl ctx(node, {input_type});

  const auto* const schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 29, "");
  ASSERT_TRUE(schema);
  EXPECT_FALSE(schema->HasFunction());
  EXPECT_TRUE(schema->HasContextDependentFunction());
  FunctionProto function;
  EXPECT_TRUE(schema->BuildContextDependentFunction(ctx, function, 29));
  EXPECT_EQ(function.name(), "MeanVarianceNormalization");
}

TEST(FunctionAPITest, ExpandMeanVarianceNormalizationDefaultEpsilon) {
  const auto* const schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 29, "");
  ASSERT_TRUE(schema);
  NodeProto node;
  node.set_op_type("MeanVarianceNormalization");
  node.add_input("X");
  node.add_output("Y");
  TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  FunctionBodyBuildContextImpl ctx(node, {input_type});

  for (const int version : {13, 18, 29}) {
    FunctionProto function;
    ASSERT_TRUE(schema->BuildContextDependentFunction(ctx, function, version));
    GraphProto expanded;
    FunctionExpandHelper(node, function, expanded);
    bool found_epsilon = false;
    for (const auto& internal_node : expanded.node()) {
      if (internal_node.op_type() == "Constant" && internal_node.output(0).find("EpsilonFloat") != std::string::npos) {
        found_epsilon = true;
        ASSERT_EQ(internal_node.attribute_size(), 1);
        const auto& value = internal_node.attribute(0);
        EXPECT_EQ(value.name(), "value_float");
        EXPECT_TRUE(value.has_f());
        EXPECT_FLOAT_EQ(value.f(), schema->attributes().at("epsilon").default_value.f());
      }
    }
    EXPECT_TRUE(found_epsilon);
  }
}

} // namespace ONNX_NAMESPACE::Test
