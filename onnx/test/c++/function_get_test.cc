#include <iostream>
#include "gtest/gtest.h"
#include "onnx/defs/function.h"

namespace ONNX_NAMESPACE {
namespace Test {
TEST(FunctionAPITest, Get_All_Functions) {
  std::multimap<std::string, std::unique_ptr<FunctionProto>> temp_map;
  FunctionBuilderRegistry& function_registry =
      FunctionBuilderRegistry::OnnxInstance();
  Common::Status status = function_registry.GetFunctions("", &temp_map);
  size_t input_size = temp_map.size();
  EXPECT_EQ(input_size, 1);
  EXPECT_EQ(temp_map.count("FuncMeanVarianceNormalization"), 1);
  auto temp_iter = temp_map.find("FuncMeanVarianceNormalization");
  EXPECT_EQ(temp_iter->second->attribute_size(), 1);
}
} // namespace Test
} // namespace ONNX_NAMESPACE