#include <iostream>
#include "driver/gtest_utils.h"

std::vector<ProtoTestCase> all_test_cases;
GTEST_DEFINE_string_(
    TARGET_DIR,
    ::testing::internal::StringFromGTestEnv("TARGET_DIR", "auto"),
    "The target repo of onnxifi data.");
GTEST_API_ int main(int argc, char** argv){
	std::cout << "Running main() from cpp test driver: cpp/test_main.cc" << std::endl;
        if (argc != 2) {
          std::cout << "target directory must be given!" << std::endl;
          return 0;
        }
        std::string target_dir = argv[1];
        all_test_cases = loadAllTestCases(target_dir);
        ::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
