// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/common/graph_shape_inference.h"

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/defs/parser.h"

namespace ONNX_NAMESPACE::Test {

static std::unique_ptr<Graph> ParseGraphIR(const char* code) {
  ModelProto model;
  OnnxParser parser(code);
  auto status = parser.Parse(model);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  return ImportModelProto(model);
}

TEST(GraphShapeInferenceTest, InfersUnknownOutputShapeFromInputs) {
  auto g = ParseGraphIR(R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 17 ]
>
agraph (float[N, 128] X, float[128, 10] W, float[10] B) => (C)
{
  T = MatMul(X, W)
  S = Add(T, B)
  C = Softmax(S)
}
)ONNX");
  ASSERT_TRUE(g != nullptr);

  EXPECT_TRUE(InferShapesOnGraph(*g));

  ASSERT_EQ(g->outputs().size(), 1u);
  const Value* c = g->outputs()[0];
  EXPECT_EQ(c->elemType(), TensorProto_DataType_FLOAT);
  ASSERT_TRUE(c->has_sizes());
  ASSERT_EQ(c->sizes().size(), 2u);
  EXPECT_FALSE(c->sizes()[0].is_int);
  EXPECT_EQ(c->sizes()[0].param, "N");
  EXPECT_TRUE(c->sizes()[1].is_int);
  EXPECT_EQ(c->sizes()[1].dim, 10);
}

TEST(GraphShapeInferenceTest, SecondRunOnConvergedGraphReportsNoChange) {
  auto g = ParseGraphIR(R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 17 ]
>
agraph (float[N, 128] X, float[128, 10] W, float[10] B) => (C)
{
  T = MatMul(X, W)
  S = Add(T, B)
  C = Softmax(S)
}
)ONNX");
  ASSERT_TRUE(g != nullptr);

  EXPECT_TRUE(InferShapesOnGraph(*g));
  EXPECT_FALSE(InferShapesOnGraph(*g));
}

TEST(GraphShapeInferenceTest, UsesInitializerValueForReshapeOutputShape) {
  auto g = ParseGraphIR(R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 17 ]
>
agraph (float[6] X) => (Y)
<int64[2] shape = {2, 3}>
{
  Y = Reshape(X, shape)
}
)ONNX");
  ASSERT_TRUE(g != nullptr);
  ASSERT_NE(g->getInitializer("shape"), nullptr);
  EXPECT_EQ(g->getInitializer("nonexistent-initializer"), nullptr);

  EXPECT_TRUE(InferShapesOnGraph(*g));

  const Value* y = g->outputs()[0];
  ASSERT_TRUE(y->has_sizes());
  ASSERT_EQ(y->sizes().size(), 2u);
  EXPECT_TRUE(y->sizes()[0].is_int);
  EXPECT_EQ(y->sizes()[0].dim, 2);
  EXPECT_TRUE(y->sizes()[1].is_int);
  EXPECT_EQ(y->sizes()[1].dim, 3);
}

TEST(GraphShapeInferenceTest, UsesConstantNodeValueForReshapeOutputShape) {
  auto g = ParseGraphIR(R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 17 ]
>
agraph (float[6] X) => (Y)
{
  shape = Constant<value = int64[2] {2, 3}>()
  Y = Reshape(X, shape)
}
)ONNX");
  ASSERT_TRUE(g != nullptr);

  EXPECT_TRUE(InferShapesOnGraph(*g));

  const Value* y = g->outputs()[0];
  ASSERT_TRUE(y->has_sizes());
  ASSERT_EQ(y->sizes().size(), 2u);
  EXPECT_TRUE(y->sizes()[0].is_int);
  EXPECT_EQ(y->sizes()[0].dim, 2);
  EXPECT_TRUE(y->sizes()[1].is_int);
  EXPECT_EQ(y->sizes()[1].dim, 3);
}

TEST(GraphShapeInferenceTest, UnknownDimsGetDistinctSymbolicNames) {
  // `shape` isn't a Constant/initializer, so Reshape only learns the output's
  // rank; each dim comes back unset and must be materialized to a distinct
  // dim_param rather than left (indistinguishably) unknown.
  auto g = ParseGraphIR(R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 17 ]
>
agraph (float[6] X, int64[2] shape) => (Y)
{
  Y = Reshape(X, shape)
}
)ONNX");
  ASSERT_TRUE(g != nullptr);

  EXPECT_TRUE(InferShapesOnGraph(*g));

  const Value* y = g->outputs()[0];
  ASSERT_TRUE(y->has_sizes());
  ASSERT_EQ(y->sizes().size(), 2u);
  for (const Dimension& d : y->sizes()) {
    EXPECT_FALSE(d.is_unknown);
    EXPECT_FALSE(d.is_int);
    EXPECT_FALSE(d.param.empty());
  }
  EXPECT_NE(y->sizes()[0].param, y->sizes()[1].param);
}

TEST(GraphShapeInferenceTest, GeneratedSymbolicNamesAvoidExistingCollisions) {
  // "unk__0" is both `already_named`'s pre-existing dim_param and the first
  // name the symbol generator would otherwise hand out (its default prefix
  // is "unk__"); newly generated names must skip over the existing one.
  auto g = ParseGraphIR(R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 17 ]
>
agraph (float[6] X, int64[2] shape, float[unk__0] already_named) => (Y)
{
  Y = Reshape(X, shape)
}
)ONNX");
  ASSERT_TRUE(g != nullptr);

  EXPECT_TRUE(InferShapesOnGraph(*g));

  const Value* y = g->outputs()[0];
  ASSERT_TRUE(y->has_sizes());
  ASSERT_EQ(y->sizes().size(), 2u);
  for (const Dimension& d : y->sizes()) {
    EXPECT_NE(d.param, "unk__0");
  }
}

TEST(GraphShapeInferenceTest, InfersThroughIfSubgraphs) {
  auto g = ParseGraphIR(R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 17 ]
>
agraph (bool cond, float[4] X, float[4] Y) => (Z)
{
  Z = If (cond) <
    then_branch = g1 () => (z_then) { z_then = Identity(X) },
    else_branch = g2 () => (z_else) { z_else = Identity(Y) }
  >
}
)ONNX");
  ASSERT_TRUE(g != nullptr);

  EXPECT_TRUE(InferShapesOnGraph(*g));

  const Value* z = g->outputs()[0];
  EXPECT_EQ(z->elemType(), TensorProto_DataType_FLOAT);
  ASSERT_TRUE(z->has_sizes());
  ASSERT_EQ(z->sizes().size(), 1u);
  EXPECT_TRUE(z->sizes()[0].is_int);
  EXPECT_EQ(z->sizes()[0].dim, 4);
}

TEST(GraphShapeInferenceTest, NodeWithNoRegisteredSchemaLeavesOutputUnchanged) {
  auto g = ParseGraphIR(R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 17, "test.custom" : 1 ]
>
agraph (float[4] X) => (Y)
{
  Y = test.custom.MysteryOp(X)
}
)ONNX");
  ASSERT_TRUE(g != nullptr);

  EXPECT_FALSE(InferShapesOnGraph(*g));

  const Value* y = g->outputs()[0];
  EXPECT_FALSE(y->has_sizes());
  EXPECT_EQ(y->elemType(), TensorProto_DataType_UNDEFINED);
}

TEST(GraphShapeInferenceTest, NodeWithNoOpsetImportForDomainLeavesOutputUnchanged) {
  // Unlike the test above, the node's domain has no opset import at all.
  auto g = ParseGraphIR(R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 17 ]
>
agraph (float[4] X) => (Y)
{
  Y = test.custom.MysteryOp(X)
}
)ONNX");
  ASSERT_TRUE(g != nullptr);

  EXPECT_FALSE(InferShapesOnGraph(*g));

  const Value* y = g->outputs()[0];
  EXPECT_FALSE(y->has_sizes());
}

} // namespace ONNX_NAMESPACE::Test
