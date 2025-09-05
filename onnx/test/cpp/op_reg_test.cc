// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "catch2/catch_test_macros.hpp"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
namespace Test {
TEST_CASE("OpRegistrationTest", "[GemmOp]") {
  auto opSchema = OpSchemaRegistry::Schema("Gemm");
  REQUIRE(nullptr != opSchema);
  size_t input_size = opSchema->inputs().size();
  REQUIRE(input_size == 3);
  REQUIRE(opSchema->inputs()[0].GetTypes() == opSchema->outputs()[0].GetTypes());
  size_t attr_size = opSchema->attributes().size();
  REQUIRE(attr_size == 4);
  REQUIRE(opSchema->attributes().count("alpha") != 0);
  REQUIRE(opSchema->attributes().at("alpha").type == AttributeProto_AttributeType_FLOAT);
  REQUIRE(opSchema->attributes().count("beta") != 0);
  REQUIRE(opSchema->attributes().at("beta").type == AttributeProto_AttributeType_FLOAT);
}
} // namespace Test
} // namespace ONNX_NAMESPACE
