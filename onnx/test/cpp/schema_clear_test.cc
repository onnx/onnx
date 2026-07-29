// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
namespace Test {

namespace {

OpSchema MakeBaseSchema() {
  OpSchema schema;
  schema.SetName("ClearTestOp")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(1)
      .SetDoc("base doc")
      .Input(0, "A", "first", "T")
      .Input(1, "B", "second", "T")
      .Input(2, "C", "third", "T")
      .Output(0, "Y", "result", "T")
      .Output(1, "Z", "extra", "T")
      .Attr("alpha", "alpha attr", AttributeProto::FLOAT, 1.0f)
      .Attr("beta", "beta attr", AttributeProto::INT, static_cast<int64_t>(0))
      .TypeConstraint("T", {"tensor(float)", "tensor(double)"}, "numeric");
  return schema;
}

} // namespace

TEST(OpSchemaClearTest, ClearAttrRemovesAndAllowsReAdd) {
  OpSchema schema = MakeBaseSchema();
  schema.ClearAttr("alpha");
  EXPECT_EQ(schema.attributes().count("alpha"), 0u);
  EXPECT_EQ(schema.attributes().count("beta"), 1u);
  schema.Attr("alpha", "alpha override", AttributeProto::INT, static_cast<int64_t>(7));
  ASSERT_EQ(schema.attributes().count("alpha"), 1u);
  EXPECT_EQ(schema.attributes().at("alpha").type, AttributeProto::INT);
}

TEST(OpSchemaClearTest, ClearTypeConstraintRemovesFromBothStorages) {
  // Type constraints live in both a map and a vector; both must be purged.
  OpSchema schema = MakeBaseSchema();
  schema.ClearTypeConstraint("T");
  EXPECT_EQ(schema.typeConstraintMap().count("T"), 0u);
  EXPECT_TRUE(schema.typeConstraintParams().empty());
}

TEST(OpSchemaClearTest, ClearTypeConstraintAllowsReAdd) {
  // TypeConstraint() throws on duplicate name; ClearTypeConstraint exists to enable override.
  OpSchema schema = MakeBaseSchema();
  schema.ClearTypeConstraint("T");
  schema.TypeConstraint("T", {"tensor(float)"}, "narrowed");
  ASSERT_EQ(schema.typeConstraintParams().size(), 1u);
  EXPECT_EQ(schema.typeConstraintParams()[0].allowed_type_strs.size(), 1u);
  EXPECT_EQ(schema.typeConstraintParams()[0].allowed_type_strs[0], "tensor(float)");
}

TEST(OpSchemaClearTest, ClearTypeConstraintMissingLeavesExistingIntact) {
  // remove_if + erase pattern must not touch unrelated entries when no match.
  OpSchema schema = MakeBaseSchema();
  schema.ClearTypeConstraint("does_not_exist");
  EXPECT_EQ(schema.typeConstraintMap().count("T"), 1u);
  ASSERT_EQ(schema.typeConstraintParams().size(), 1u);
  EXPECT_EQ(schema.typeConstraintParams()[0].type_param_str, "T");
}

TEST(OpSchemaClearTest, TruncateInputsFromThenFinalizeRecomputesMinMax) {
  OpSchema schema = MakeBaseSchema();
  schema.TruncateInputsFrom(1);
  ASSERT_EQ(schema.inputs().size(), 1u);
  schema.Finalize();
  EXPECT_EQ(schema.min_input(), 1);
  EXPECT_EQ(schema.max_input(), 1);
}

TEST(OpSchemaClearTest, TruncateOutputsFromThenFinalizeRecomputesMinMax) {
  OpSchema schema = MakeBaseSchema();
  schema.TruncateOutputsFrom(1);
  ASSERT_EQ(schema.outputs().size(), 1u);
  schema.Finalize();
  EXPECT_EQ(schema.min_output(), 1);
  EXPECT_EQ(schema.max_output(), 1);
}

TEST(OpSchemaClearTest, ClearDocClearsAndAllowsReSet) {
#ifndef __ONNX_NO_DOC_STRINGS
  OpSchema schema = MakeBaseSchema();
  schema.ClearDoc();
  // doc() returns nullptr when the internal string is empty.
  EXPECT_EQ(schema.doc(), nullptr);
  schema.SetDoc("new doc");
  ASSERT_NE(schema.doc(), nullptr);
  EXPECT_EQ(std::string(schema.doc()), "new doc");
#endif
}

TEST(OpSchemaClearTest, InheritThenOverrideTypeConstraintRoundTrip) {
  // End-to-end pattern mirroring math/old.cc v13: a base FillUsing builder
  // sets the schema, then the caller narrows one type constraint.
  auto base_builder = [](OpSchema& s) {
    s.SetName("DeltaOp")
        .Input(0, "X", "in", "T")
        .Output(0, "Y", "out", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(double)"}, "wide");
  };
  OpSchema schema;
  schema.FillUsing(base_builder).ClearTypeConstraint("T").TypeConstraint("T", {"tensor(float)"}, "narrow");

  ASSERT_EQ(schema.typeConstraintParams().size(), 1u);
  EXPECT_EQ(schema.typeConstraintParams()[0].allowed_type_strs.size(), 1u);
  EXPECT_EQ(schema.typeConstraintParams()[0].allowed_type_strs[0], "tensor(float)");
  EXPECT_EQ(schema.typeConstraintParams()[0].description, "narrow");
}

} // namespace Test
} // namespace ONNX_NAMESPACE
