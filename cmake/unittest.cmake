# SPDX-License-Identifier: Apache-2.0

set(UT_NAME ${PROJECT_NAME}_gtests)
set(ONNX_ROOT ${PROJECT_SOURCE_DIR})

include(${ONNX_ROOT}/cmake/Utils.cmake)
include(CTest)

set(${UT_NAME}_libs ${googletest_STATIC_LIBRARIES})

list(APPEND ${UT_NAME}_libs onnx onnx_proto)

file(GLOB_RECURSE ${UT_NAME}_src "${ONNX_ROOT}/onnx/test/cpp/*.cc")
find_package(Threads REQUIRED)

function(AddTest)
  cmake_parse_arguments(_UT "" "TARGET" "LIBS;SOURCES" ${ARGN})

  list(REMOVE_DUPLICATES _UT_LIBS)
  list(REMOVE_DUPLICATES _UT_SOURCES)

  add_executable(${_UT_TARGET} ${_UT_SOURCES})

  target_include_directories(${_UT_TARGET} PUBLIC ${ONNX_INCLUDE_DIRS})
  target_link_libraries(${_UT_TARGET} ${_UT_LIBS} Threads::Threads)

  if(MSVC)
    add_msvc_runtime_flag(${_UT_TARGET})
    target_compile_options(${_UT_TARGET}
                           PRIVATE /wd4146 # unary minus operator applied to
                                           # unsigned type, result still
                                           # unsigned from include\google\protob
                                           # uf\wire_format_lite.h
                                 /wd4244 # 'argument': conversion from 'google::
                                         # protobuf::uint64' to 'int', possible
                                         # loss of data
                                 /wd4267 # Conversion from 'size_t' to 'int',
                                         # possible loss of data
                                 /wd4996 # The second parameter is ignored.
                           )
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

addtest(TARGET ${UT_NAME} SOURCES ${${UT_NAME}_src} LIBS ${${UT_NAME}_libs})
