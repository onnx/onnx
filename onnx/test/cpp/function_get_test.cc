// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "catch2/catch_test_macros.hpp"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
namespace Test {

TEST_CASE("FunctionAPITest", "[GetFunctionOpWithVersion]") {
  const auto* schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 9, "");
  REQUIRE(schema);
  REQUIRE(schema->HasFunction());
  auto func = schema->GetFunction();
  REQUIRE(func->name() == "MeanVarianceNormalization");
}

TEST_CASE("FunctionAPITest", "[GetMeanVarianceNormalizationFunctionWithVersion]") {
  {
    const auto* schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 13, "");
    REQUIRE(schema);
    REQUIRE(schema->HasFunction());
    auto func = schema->GetFunction();
    REQUIRE(func->name() == "MeanVarianceNormalization");
  }
  {
    const auto* schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 17, "");
    REQUIRE(schema);
    REQUIRE(schema->HasFunction());
    auto func = schema->GetFunction();
    REQUIRE(func->name() == "MeanVarianceNormalization");
  }
  {
    const auto* schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 18, "");
    REQUIRE(schema);
    REQUIRE(schema->HasFunction());
    auto func = schema->GetFunction();
    REQUIRE(func->name() == "MeanVarianceNormalization");
  }
}

} // namespace Test
} // namespace ONNX_NAMESPACE
