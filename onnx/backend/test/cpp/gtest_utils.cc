#include "gtest_utils.h"
namespace ONNX_NAMESPACE {
namespace testing {
std::vector<ONNX_NAMESPACE::testing::ResolvedTestCase>& GetTestCases() {
  static std::vector<ONNX_NAMESPACE::testing::ResolvedTestCase> all_test_cases;
  return all_test_cases;
}
} // namespace testing
} // namespace ONNX_NAMESPACE
