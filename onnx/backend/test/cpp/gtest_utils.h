#include "gtest/gtest.h"
#include "onnx/backend/test/cpp/driver/test_driver.h"
class OnnxifiGtestWrapper {
 public:
  static std::vector<ONNX_NAMESPACE::testing::ProtoTestCase> all_test_cases;
  static std::vector<ONNX_NAMESPACE::testing::ProtoTestCase> GetTestCases();
  static void SetTestCases(
      std::vector<ONNX_NAMESPACE::testing::ProtoTestCase>&);
};
