#include <iostream>
#include "gtest/gtest.h"
#include "onnx/common/constants.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
namespace Test {

TEST(FunctionAPITest, Get_Function_op_With_Version) {
  const auto* schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 9, "");
  EXPECT_TRUE(schema);
  EXPECT_TRUE(schema->HasFunction());
  auto func = schema->GetFunction();
  EXPECT_EQ(func->name(), "MeanVarianceNormalization");
}

} // namespace Test
} // namespace ONNX_NAMESPACE