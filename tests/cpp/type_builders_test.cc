// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE::Test {

namespace t = ONNX_NAMESPACE::types;

TEST(TypeBuildersTest, Factories) {
  EXPECT_EQ(t::Tensor(TensorProto::FLOAT), t::Float);
  EXPECT_EQ(t::SparseTensor(TensorProto::FLOAT), "sparse_tensor(float)");
  EXPECT_EQ(t::Sequence(t::Float), "seq(tensor(float))");
  EXPECT_EQ(t::Optional(t::Sequence(t::Double)), "optional(seq(tensor(double)))");
}

TEST(TypeBuildersTest, MapConvention) {
  // Space after the comma; bare-scalar value form for ai.onnx.ml schemas.
  EXPECT_EQ(t::Map<TensorProto::INT64>(t::Float), "map(int64, tensor(float))");
  EXPECT_EQ(t::Map<TensorProto::INT64>("float"), "map(int64, float)");
}

// Invalid map keys are rejected at compile time via static_assert; uncomment to
// verify it fails to compile:
//   t::Map<TensorProto::FLOAT>(t::Int64);
//   t::Map<TensorProto::BOOL>(t::Int64);

TEST(TypeBuildersTest, VectorHelpers) {
  EXPECT_EQ(
      t::Tensors({TensorProto::FLOAT, TensorProto::INT64}),
      (std::vector<std::string>{"tensor(float)", "tensor(int64)"}));
  EXPECT_EQ(t::SparseTensors({TensorProto::FLOAT}), (std::vector<std::string>{"sparse_tensor(float)"}));
  EXPECT_EQ(
      t::Sequence(std::vector<std::string>{t::Float, t::Int64}),
      (std::vector<std::string>{"seq(tensor(float))", "seq(tensor(int64))"}));
  EXPECT_EQ(t::Optional(std::vector<std::string>{t::Bool}), (std::vector<std::string>{"optional(tensor(bool))"}));
  EXPECT_EQ(t::Concat(std::vector<std::string>{"a", "b"}, {"c"}), (std::vector<std::string>{"a", "b", "c"}));
}

TEST(TypeBuildersTest, DropsIntoOpSchema) {
  OpSchema schema;
  schema.TypeConstraint("T", {t::Float, t::Sequence(t::Float), t::Optional(t::Int64)}, "test");
  const auto& tcp = schema.typeConstraintParams().front();
  ASSERT_EQ(tcp.allowed_type_strs.size(), 3u);
  EXPECT_EQ(tcp.allowed_type_strs[0], "tensor(float)");
  EXPECT_EQ(tcp.allowed_type_strs[1], "seq(tensor(float))");
  EXPECT_EQ(tcp.allowed_type_strs[2], "optional(tensor(int64))");
}

} // namespace ONNX_NAMESPACE::Test
