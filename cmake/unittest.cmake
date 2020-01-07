set(UT_NAME ${PROJECT_NAME}_gtests)
set(ONNX_ROOT ${PROJECT_SOURCE_DIR})
set(ONNXIFI_TEST_DRIVER onnxifi_test_driver_gtests)

include(${ONNX_ROOT}/cmake/Utils.cmake)

find_package(Threads)

set(${UT_NAME}_libs ${googletest_STATIC_LIBRARIES})
set(${ONNXIFI_TEST_DRIVER}_libs ${googletest_STATIC_LIBRARIES})

list(APPEND ${UT_NAME}_libs onnx)
list(APPEND ${UT_NAME}_libs onnx_proto)
list(APPEND ${UT_NAME}_libs onnxifi_loader)
list(APPEND ${UT_NAME}_libs onnxifi)
list(APPEND ${UT_NAME}_libs ${PROTOBUF_LIBRARIES})

list(APPEND ${ONNXIFI_TEST_DRIVER}_libs onnx)
list(APPEND ${ONNXIFI_TEST_DRIVER}_libs onnx_proto)
list(APPEND ${ONNXIFI_TEST_DRIVER}_libs onnxifi_loader)
list(APPEND ${ONNXIFI_TEST_DRIVER}_libs ${PROTOBUF_LIBRARIES})
list(APPEND ${ONNXIFI_TEST_DRIVER}_libs onnxifi)

file(GLOB_RECURSE ${UT_NAME}_src "${ONNX_ROOT}/onnx/test/cpp/*.cc")
file(GLOB_RECURSE ${ONNXIFI_TEST_DRIVER}_src "${ONNX_ROOT}/onnx/backend/test/cpp/*.h" "${ONNX_ROOT}/onnx/backend/test/cpp/*.cc")

function(AddTest)
  cmake_parse_arguments(_UT "" "TARGET" "LIBS;SOURCES" ${ARGN})

  list(REMOVE_DUPLICATES _UT_LIBS)
  list(REMOVE_DUPLICATES _UT_SOURCES)

  add_executable(${_UT_TARGET} ${_UT_SOURCES})
  add_dependencies(${_UT_TARGET} onnx onnx_proto googletest)

  target_include_directories(${_UT_TARGET}
                             PUBLIC ${googletest_INCLUDE_DIRS}
                                    ${ONNX_INCLUDE_DIRS}
                                    ${PROTOBUF_INCLUDE_DIRS}
                                    ${ONNX_ROOT}
                                    ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(${_UT_TARGET} ${_UT_LIBS} ${CMAKE_THREAD_LIBS_INIT})
  if(TARGET protobuf::libprotobuf)
    target_link_libraries(${_UT_TARGET} protobuf::libprotobuf)
  else()
    target_link_libraries(${_UT_TARGET} ${PROTOBUF_LIBRARIES})
  endif()

  if(WIN32)
    target_compile_options(${_UT_TARGET}
                           PRIVATE /EHsc # exception handling - C++ may throw,
                                         # extern "C" will not
                           )
    add_msvc_runtime_flag(${_UT_TARGET})
  endif()

  if(MSVC)
    target_compile_options(${_UT_TARGET}
                           PRIVATE /wd4146 # unary minus operator applied to
                                           # unsigned type, result still
                                           # unsigned from include\google\protob
                                           # uf\wire_format_lite.h
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
addtest(TARGET ${ONNXIFI_TEST_DRIVER} SOURCES ${${ONNXIFI_TEST_DRIVER}_src} LIBS ${${ONNXIFI_TEST_DRIVER}_libs})
