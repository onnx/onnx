#include "gtest_utils.h"
std::vector<ONNX_NAMESPACE::testing::ProtoTestCase>
    OnnxifiGtestWrapper::all_test_cases;
std::vector<ONNX_NAMESPACE::testing::ProtoTestCase>
OnnxifiGtestWrapper::GetTestCases() {
  return OnnxifiGtestWrapper::all_test_cases;
}

void OnnxifiGtestWrapper::SetTestCases(
    std::vector<ONNX_NAMESPACE::testing::ProtoTestCase>& t) {
  OnnxifiGtestWrapper::all_test_cases = t;
}
