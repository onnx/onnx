#include "gtest_utils.h"
std::vector<ONNX_NAMESPACE::testing::ResolvedTestCase>& GetTestCases() {
  static std::vector<ONNX_NAMESPACE::testing::ResolvedTestCase> all_test_cases;
  return all_test_cases;
}
