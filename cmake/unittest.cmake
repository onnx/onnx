set(UT_NAME ${PROJECT_NAME}_gtests)
set(ONNX_ROOT ${PROJECT_SOURCE_DIR})

include(${ONNX_ROOT}/cmake/Utils.cmake)

find_package(Threads)

function(add_whole_archive_flag lib output_var)
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    set(${output_var} -Wl,-force_load,$<TARGET_FILE:${lib}> PARENT_SCOPE)
  elseif(MSVC)
    # In MSVC, we will add whole archive in default.
    set(${output_var} -WHOLEARCHIVE:$<SHELL_PATH:$<TARGET_FILE:${lib}>>
        PARENT_SCOPE)
  else()
    # Assume everything else is like gcc
    set(${output_var}
        "-Wl,--whole-archive $<TARGET_FILE:${lib}> -Wl,--no-whole-archive"
        PARENT_SCOPE)
  endif()
endfunction()

set(${UT_NAME}_libs ${googletest_STATIC_LIBRARIES})

add_whole_archive_flag(onnx tmp)
list(APPEND ${UT_NAME}_libs ${tmp})
list(APPEND ${UT_NAME}_libs onnx_proto)
list(APPEND ${UT_NAME}_libs onnxifi_loader)
list(APPEND ${UT_NAME}_libs ${PROTOBUF_LIBRARIES})

file(GLOB_RECURSE ${UT_NAME}_src "${ONNX_ROOT}/onnx/test/cpp/*.cc")

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

set(TEST_DATA_SRC ${ONNX_ROOT}/onnx/test/cpp/testdata)
set(TEST_DATA_DES $<TARGET_FILE_DIR:${UT_NAME}>/testdata)

# Copy test data from source to destination.
add_custom_command(TARGET ${UT_NAME} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_DATA_SRC}
                           ${TEST_DATA_DES})
