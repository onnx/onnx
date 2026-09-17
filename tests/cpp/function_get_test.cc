// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
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
  NodeProto node;
  node.set_op_type("MeanVarianceNormalization");
  node.add_input("X");
  node.add_output("Y");
  TypeProto input_type;
  input_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  FunctionBodyBuildContextImpl ctx(node, {input_type});

  for (const int version : {13, 17, 18, 28, 29}) {
    const auto* const schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", version, "");
    ASSERT_TRUE(schema);
    EXPECT_TRUE(schema->HasContextDependentFunction());
    FunctionProto function;
    EXPECT_TRUE(schema->BuildContextDependentFunction(ctx, function, version));
    EXPECT_EQ(function.name(), "MeanVarianceNormalization");
  }
}

} // namespace ONNX_NAMESPACE::Test
