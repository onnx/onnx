#include "gtest_utils.h"
std::vector<ONNX_NAMESPACE::testing::ProtoTestCase>& GetTestCases() {
  static std::vector<ONNX_NAMESPACE::testing::ProtoTestCase> all_test_cases;
  return all_test_cases;
}
