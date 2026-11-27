# SPDX-License-Identifier: Apache-2.0

include(CTest)

set(ONNX_ROOT ${PROJECT_SOURCE_DIR})
set(UT_NAME ${PROJECT_NAME}_gtests)
set(test_src
    ${ONNX_ROOT}/tests/cpp/checker_test.cc
    ${ONNX_ROOT}/tests/cpp/data_propagation_test.cc
    ${ONNX_ROOT}/tests/cpp/function_context_test.cc
    ${ONNX_ROOT}/tests/cpp/function_get_test.cc
    ${ONNX_ROOT}/tests/cpp/function_verify_test.cc
    ${ONNX_ROOT}/tests/cpp/graph_shape_inference_test.cc
    ${ONNX_ROOT}/tests/cpp/inliner_test.cc
    ${ONNX_ROOT}/tests/cpp/ir_test.cc
    ${ONNX_ROOT}/tests/cpp/op_reg_test.cc
    ${ONNX_ROOT}/tests/cpp/parser_test.cc
    ${ONNX_ROOT}/tests/cpp/safe_math_test.cc
    ${ONNX_ROOT}/tests/cpp/schema_registration_test.cc
    ${ONNX_ROOT}/tests/cpp/shape_inference_test.cc
    ${ONNX_ROOT}/tests/cpp/tensor_test.cc
    ${ONNX_ROOT}/tests/cpp/test_main.cc
    ${ONNX_ROOT}/tests/cpp/utf8_conversion_test.cc
    ${ONNX_ROOT}/tests/cpp/version_converter_test.cc
)
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
