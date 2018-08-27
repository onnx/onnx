#include <iostream>
#include "driver/gtest_utils.h"

std::vector<onnx::testing::ProtoTestCase> all_test_cases;
GTEST_API_ int main(int argc, char** argv){
        if (argc != 2) {
          std::cout << "target directory must be given!" << std::endl;
          return 0;
        }
        std::string target_dir = argv[1];
        all_test_cases = onnx::testing::LoadAllTestCases(target_dir);
        ::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
