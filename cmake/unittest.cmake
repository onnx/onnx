# SPDX-License-Identifier: Apache-2.0

set(UT_NAME ${PROJECT_NAME}_gtests)
set(ONNX_ROOT ${PROJECT_SOURCE_DIR})

include(${ONNX_ROOT}/cmake/Utils.cmake)
include(CTest)

file(GLOB_RECURSE ${UT_NAME}_src "${ONNX_ROOT}/onnx/test/cpp/*.cc")
find_package(Threads REQUIRED)

function(AddTest)
  cmake_parse_arguments(_UT "" "TARGET" "SOURCES" ${ARGN})

  list(REMOVE_DUPLICATES _UT_SOURCES)

  add_executable(${_UT_TARGET} ${_UT_SOURCES})

  target_include_directories(${_UT_TARGET} PUBLIC ${ONNX_INCLUDE_DIRS})
  target_link_libraries(${_UT_TARGET} gtest_main onnx onnx_proto Threads::Threads)

  if(MSVC)
    add_msvc_runtime_flag(${_UT_TARGET})
  endif()

  set(TEST_ARGS)
  if(ONNX_GENERATE_TEST_REPORTS)
    # generate a report file next to the test program
    list(
      APPEND
        TEST_ARGS
        "--gtest_output=xml:$<SHELL_PATH:$<TARGET_FILE:${_UT_TARGET}>.$<CONFIG>.results.xml>"
      )
  endif()

  add_test(NAME ${_UT_TARGET}
           COMMAND ${_UT_TARGET} ${TEST_ARGS}
           WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>)

endfunction(AddTest)

addtest(TARGET ${UT_NAME} SOURCES ${${UT_NAME}_src})
