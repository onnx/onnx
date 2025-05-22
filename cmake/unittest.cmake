# SPDX-License-Identifier: Apache-2.0

include(CTest)

set(ONNX_ROOT ${PROJECT_SOURCE_DIR})
set(UT_NAME ${PROJECT_NAME}_gtests)
file(GLOB_RECURSE test_src "${ONNX_ROOT}/onnx/test/cpp/*.cc")
add_executable(${UT_NAME} ${test_src})
find_package(Threads REQUIRED)
target_link_libraries(${UT_NAME} PRIVATE onnx Threads::Threads)
if(TARGET GTest::gtest)
  target_link_libraries(${UT_NAME} PRIVATE GTest::gtest)
else()
  target_link_libraries(${UT_NAME} PRIVATE gtest)
endif()

set(TEST_ARGS)
if(ONNX_GENERATE_TEST_REPORTS)
  # generate a report file next to the test program
  list(
      APPEND
        TEST_ARGS
        "--gtest_output=xml:$<SHELL_PATH:$<TARGET_FILE:${UT_NAME}>.$<CONFIG>.results.xml>")
endif()

add_test(NAME ${UT_NAME} COMMAND ${UT_NAME} ${TEST_ARGS})
