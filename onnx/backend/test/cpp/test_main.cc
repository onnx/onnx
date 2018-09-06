#include <iostream>
#include "driver/gtest_utils.h"

std::vector<ONNX_NAMESPACE::testing::ProtoTestCase> all_test_cases;
GTEST_API_ int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "target directory must be given!" << std::endl;
    return EXIT_FAILURE;
  }
  std::string target_dir = argv[1];
  all_test_cases = ONNX_NAMESPACE::testing::LoadAllTestCases(target_dir);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
