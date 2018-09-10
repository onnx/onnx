#include <iostream>
#include "gtest_utils.h"

GTEST_API_ int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "target directory must be given!" << std::endl;
    return EXIT_FAILURE;
  }
  std::string target_dir = argv[1];
  auto testcases = ONNX_NAMESPACE::testing::LoadAllTestCases(target_dir);
  GetTestCases() = testcases;
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
