# Minimum CMake required
cmake_minimum_required(VERSION 3.24)
# Project
project(onnx LANGUAGES CXX)

include(cmake/Utils.cmake)
# Honor visibility properties for all target types.
cmake_policy(SET CMP0063 NEW)

# Set default build type
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT isMultiConfig AND NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(
    CMAKE_BUILD_TYPE "Release"
    CACHE
      STRING
      "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage."
    FORCE)
endif()

# https://reproducible-builds.org/docs/source-date-epoch/
if(DEFINED ENV{SOURCE_DATE_EPOCH})
  execute_process(
    COMMAND "date" "-u" "-d" "@$ENV{SOURCE_DATE_EPOCH}" "+%Y-%m-%d"
    OUTPUT_VARIABLE BUILD_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
  execute_process(
    COMMAND "date" "+%Y-%m-%d"
    OUTPUT_VARIABLE BUILD_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

if(NOT BUILD_SHARED_LIBS)
  # by default, cmake builds static libraries
  set(BUILD_SHARED_LIBS OFF)
endif()


if(DEFINED BUILD_ONNX_PYTHON)
  message(WARNING "'BUILD_ONNX_PYTHON' is deprecated. Please, use 'ONNX_BUILD_PYTHON' instead")
  set(ONNX_BUILD_PYTHON_DEFAULT ${BUILD_ONNX_PYTHON})
else()
  set(ONNX_BUILD_PYTHON_DEFAULT OFF)
endif()

option(ONNX_BUILD_PYTHON "Build Python binaries" ${ONNX_BUILD_PYTHON_DEFAULT})
option(ONNX_BUILD_CUSTOM_PROTOBUF "Build and use ONNX's own protobuf" OFF)
option(ONNX_USE_PROTOBUF_SHARED_LIBS "Build ONNX using protobuf shared library." OFF)
option(ONNX_GEN_PB_TYPE_STUBS "Generate protobuf python type stubs" ON)
option(ONNX_WERROR "Build with Werror" OFF)
option(ONNX_COVERAGE "Build with coverage instrumentation" OFF)
option(ONNX_BUILD_TESTS "Build ONNX C++ APIs Tests" OFF)
option(ONNX_USE_ASAN "Build ONNX with ASAN" OFF)
option(ONNX_USE_LITE_PROTO "Use lite protobuf instead of full." OFF)
option(ONNX_DISABLE_EXCEPTIONS "Disable exception handling." OFF)
option(ONNX_DISABLE_STATIC_REGISTRATION "Disable static registration for ONNX operator schemas." OFF)
option(ONNX_USE_UNITY_BUILD "Enable Unity (Jumbo) build for" OFF)
if(WIN32)
  option(ONNX_USE_MSVC_STATIC_RUNTIME "Build with MSVC static runtime" OFF)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED ONNX_ML)
  if(DEFINED ENV{ONNX_ML})
    set(DEFAULT_ONNX_ML $ENV{ONNX_ML})
  else()
    set(DEFAULT_ONNX_ML ON)
  endif()
  option(ONNX_ML "Enable traditional ML API." ${DEFAULT_ONNX_ML})
endif()

if(NOT DEFINED ONNX_VERIFY_PROTO3)
  if(DEFINED ENV{ONNX_VERIFY_PROTO3})
    set(PROTO3_ENABLED $ENV{ONNX_VERIFY_PROTO3})
  else()
    set(PROTO3_ENABLED OFF)
  endif()
  option(ONNX_VERIFY_PROTO3 "Generate code by proto3" ${PROTO3_ENABLED})
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
else()
  if(CMAKE_CXX_STANDARD VERSION_LESS 17)
    message(FATAL_ERROR "At least C++17 is required.")
  endif()
endif()

include(GNUInstallDirs)

set(ONNX_ROOT ${onnx_SOURCE_DIR})

# Read ONNX version
file(READ "${ONNX_ROOT}/VERSION_NUMBER" ONNX_VERSION)
string(STRIP "${ONNX_VERSION}" ONNX_VERSION)

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnon-virtual-dtor")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  if(ONNX_COVERAGE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  endif()
endif()

if(NOT ONNX_NAMESPACE)
  set(ONNX_NAMESPACE "onnx")
endif()

if(MSVC)
  if(NOT ONNX_DISABLE_EXCEPTIONS)
    string(APPEND CMAKE_CXX_FLAGS " /EHsc /wd26812")
    string(APPEND CMAKE_C_FLAGS " /EHsc /wd26812")
  endif()
  add_compile_options(/MP /utf-8 /nologo)
  # For disabling Protobuf related warnings
  add_compile_options(/wd4146 # unary minus operator applied to unsigned type,
                              # result still unsigned
                      /wd4244 # 'argument': conversion from 'google::
                              # protobuf::uint64' to 'int', possible
                              # loss of data
                      /wd4267 # Conversion from 'size_t' to 'int',
                              # possible loss of data
                      /wd4141 # 'inline': used more than once
  )
endif()

if(ONNX_DISABLE_EXCEPTIONS)
  add_compile_definitions("ONNX_NO_EXCEPTIONS")
  # Disable C++ exceptions.
  if(MSVC)
    string(REGEX REPLACE "/EHsc" "/EHs-c-" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    add_definitions(-D_HAS_EXCEPTIONS=0)
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables")
  endif()
endif()

if(ONNX_BUILD_PYTHON)
  set(python_dev_component Development.Module)
endif()

# explicitly configure FindPython3.cmake to find python3 in virtual environment first
if(NOT DEFINED Python3_FIND_VIRTUALENV)
  set(Python3_FIND_VIRTUALENV FIRST)
endif()

if(NOT DEFINED Python3_FIND_STRATEGY)
  set(Python3_FIND_STRATEGY LOCATION)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter ${python_dev_component})

if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
  set(CMAKE_NO_SYSTEM_FROM_IMPORTED 1)
endif()

# Build the libraries with -fPIC including the protobuf lib.
if(NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

list(APPEND CMAKE_MODULE_PATH ${ONNX_ROOT}/cmake/external)
if(NOT ONNX_BUILD_CUSTOM_PROTOBUF)
  if((ONNX_USE_LITE_PROTO AND TARGET protobuf::libprotobuf-lite) OR ((NOT ONNX_USE_LITE_PROTO) AND TARGET protobuf::libprotobuf))
    # Sometimes we need to use protoc compiled for host architecture while linking
    # libprotobuf against target architecture. See https://github.com/caffe2/caffe
    # 2/blob/96f35ad75480b25c1a23d6e9e97bccae9f7a7f9c/cmake/ProtoBuf.cmake#L92-L99
    if(EXISTS "${ONNX_CUSTOM_PROTOC_EXECUTABLE}")
      message(STATUS "Using custom protoc executable")
      set(ONNX_PROTOC_EXECUTABLE ${ONNX_CUSTOM_PROTOC_EXECUTABLE})
    else()
      if(TARGET protobuf::protoc)
        set(ONNX_PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
      endif()
    endif()
  else()
    # Customized version of find Protobuf. We need to avoid situations mentioned
    # in https://github.com/caffe2/caffe2/blob/b7d983f255ef5496474f1ea188edb5e0ac4
    # 42761/cmake/ProtoBuf.cmake#L82-L92 The following section is stolen from
    # cmake/ProtoBuf.cmake in Caffe2
    find_program(Protobuf_PROTOC_EXECUTABLE
                NAMES protoc
                DOC "The Google Protocol Buffers Compiler")

    # Only if protoc was found, seed the include directories and libraries. We
    # assume that protoc is installed at PREFIX/bin. We use get_filename_component
    # to resolve PREFIX.
    if(Protobuf_PROTOC_EXECUTABLE)
      set(ONNX_PROTOC_EXECUTABLE ${Protobuf_PROTOC_EXECUTABLE})
      get_filename_component(_PROTOBUF_INSTALL_PREFIX
                            ${Protobuf_PROTOC_EXECUTABLE} DIRECTORY)
      get_filename_component(_PROTOBUF_INSTALL_PREFIX
                            ${_PROTOBUF_INSTALL_PREFIX}/.. REALPATH)
      find_library(Protobuf_PROTOC_LIBRARY
                  NAMES protoc
                  PATHS ${_PROTOBUF_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
                  NO_DEFAULT_PATH)
      if(ONNX_USE_LITE_PROTO)
        find_library(Protobuf_LITE_LIBRARY
          NAMES protobuf-lite
          PATHS ${_PROTOBUF_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
          NO_DEFAULT_PATH)
      else()
        find_library(Protobuf_LIBRARY
          NAMES protobuf
          PATHS ${_PROTOBUF_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
          NO_DEFAULT_PATH)
      endif(ONNX_USE_LITE_PROTO)
      find_path(Protobuf_INCLUDE_DIR google/protobuf/service.h
                PATHS ${_PROTOBUF_INSTALL_PREFIX}/include
                NO_DEFAULT_PATH)
      if(ONNX_USE_PROTOBUF_SHARED_LIBS)
        set(Protobuf_USE_STATIC_LIBS OFF)
      else()
        set(Protobuf_USE_STATIC_LIBS ON)
      endif()
      find_package(Protobuf)
      if(Protobuf_FOUND)
        set(PROTOBUF_DIR "${_PROTOBUF_INSTALL_PREFIX}")
        set(Build_Protobuf OFF)
        if("${Protobuf_VERSION}" VERSION_GREATER_EQUAL "4.22.0")
          # There are extra dependencies for protobuf.
          find_package(absl REQUIRED)
          find_package(utf8_range)
          message(STATUS "absl_VERSION: ${absl_VERSION}")
          set(protobuf_ABSL_USED_TARGETS
            absl::absl_check
            absl::absl_log
            absl::algorithm
            absl::base
            absl::bind_front
            absl::bits
            absl::btree
            absl::cleanup
            absl::cord
            absl::core_headers
            absl::debugging
            absl::die_if_null
            absl::dynamic_annotations
            absl::flags
            absl::flat_hash_map
            absl::flat_hash_set
            absl::function_ref
            absl::hash
            absl::layout
            absl::log_initialize
            absl::log_severity
            absl::memory
            absl::node_hash_map
            absl::node_hash_set
            absl::optional
            absl::span
            absl::status
            absl::statusor
            absl::strings
            absl::synchronization
            absl::time
            absl::type_traits
            absl::utility
            absl::variant
            utf8_range::utf8_range
            utf8_range::utf8_validity
          )
        endif()
      endif()
    endif()
  endif()
endif()
if(NOT ONNX_PROTOC_EXECUTABLE)
    set(Build_Protobuf ON)
    set(protobuf_MSVC_STATIC_RUNTIME ${ONNX_USE_MSVC_STATIC_RUNTIME})

    include(FetchContent)
    set(ABSL_PROPAGATE_CXX_STD 1)
    set(ONNX_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(ONNX_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    # Use this setting to build third-party libs.
    set(BUILD_SHARED_LIBS ${ONNX_USE_PROTOBUF_SHARED_LIBS})
    set(ProtobufURL https://github.com/protocolbuffers/protobuf/releases/download/v31.0/protobuf-31.0.tar.gz)
    set(ProtobufSHA1 c0cecb7cfc881f56a5b09f00a965aae408265fd0)
    FetchContent_Declare(
      Protobuf
      URL ${ProtobufURL}
      URL_HASH SHA1=${ProtobufSHA1}
    )
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build protobuf tests" FORCE)
    message(STATUS "Download and build Protobuf from ${ProtobufURL}")
    FetchContent_MakeAvailable(Protobuf)
    set(ONNX_PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
    set(Protobuf_VERSION "6.31.0")
    # Change back the BUILD_SHARED_LIBS to control the onnx project.
    set(BUILD_SHARED_LIBS ${ONNX_BUILD_SHARED_LIBS})
    set(PROTOBUF_DIR "${protobuf_BINARY_DIR}")
    set(CMAKE_CXX_FLAGS ${ONNX_CMAKE_CXX_FLAGS})
endif()
message(STATUS "ONNX_PROTOC_EXECUTABLE: ${ONNX_PROTOC_EXECUTABLE}")

# function(RELATIVE_PROTOBUF_GENERATE_CPP SRCS HDRS ROOT_DIR) from https://githu
# b.com/tensorflow/tensorflow/blob/d2c3b873c6f8ff999a2e4ee707a84ff00d9c15a5/tens
# orflow/contrib/cmake/tf_core_framework.cmake to solve the problem that
# customized dir can't be specified when calling PROTOBUF_GENERATE_CPP.
function(RELATIVE_PROTOBUF_GENERATE_CPP SRCS)
  if(NOT ARGN)
    message(
      SEND_ERROR
        "Error: RELATIVE_PROTOBUF_GENERATE_CPP() called without any proto files"
      )
    return()
  endif()

  set(${SRCS})

  set(GEN_PROTO_PY "${ONNX_ROOT}/onnx/gen_proto.py")
  set(GENERATED_FILE_TARGETS)
  foreach(INFILE ${ARGN})
    set(ABS_FILE "${ONNX_ROOT}/${INFILE}")
    get_filename_component(FILE_DIR ${ABS_FILE} DIRECTORY)
    get_filename_component(FILE_WE ${INFILE} NAME_WE)
    # "onnx-data" check is because we do not want to create/compile an "onnx-data-ml.proto" file
    if(ONNX_ML AND NOT(FILE_WE STREQUAL "onnx-data"))
      if(ONNX_NAMESPACE STREQUAL "onnx")
        set(GENERATED_FILE_WE "${FILE_WE}-ml")
      else()
        set(GENERATED_FILE_WE "${FILE_WE}_${ONNX_NAMESPACE}-ml")
      endif()
    else()
      if(ONNX_NAMESPACE STREQUAL "onnx")
        set(GENERATED_FILE_WE "${FILE_WE}")
      else()
        set(GENERATED_FILE_WE "${FILE_WE}_${ONNX_NAMESPACE}")
      endif()
    endif()
    file(RELATIVE_PATH REL_DIR "${ONNX_ROOT}" "${FILE_DIR}")
    set(OUTPUT_PROTO_DIR "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}")

    set(OUTPUT_PB_SRC "${OUTPUT_PROTO_DIR}/${GENERATED_FILE_WE}.pb.cc")
    set(GENERATED_PROTO "${OUTPUT_PROTO_DIR}/${GENERATED_FILE_WE}.proto")
    list(APPEND ${SRCS} "${OUTPUT_PB_SRC}")

    if(NOT EXISTS "${OUTPUT_PROTO_DIR}")
      file(MAKE_DIRECTORY "${OUTPUT_PROTO_DIR}")
    endif()

    set(GEN_PROTO_ARGS
        -p
        "${ONNX_NAMESPACE}"
        -o
        "${OUTPUT_PROTO_DIR}"
        "${FILE_WE}")
    if(ONNX_ML)
      list(APPEND GEN_PROTO_ARGS -m)
    endif()
    if(ONNX_USE_LITE_PROTO)
      list(APPEND GEN_PROTO_ARGS -l)
    endif()
    if(ONNX_VERIFY_PROTO3)
        if(NOT ONNX_PROTOC_EXECUTABLE)
          message(FATAL_ERROR "Protobuf compiler not found")
        endif()
        list(APPEND GEN_PROTO_ARGS --protoc_path)
        list(APPEND GEN_PROTO_ARGS "${ONNX_PROTOC_EXECUTABLE}")
    endif()

    add_custom_target("${GENERATED_FILE_WE}_proto_file"
                       COMMAND Python3::Interpreter "${GEN_PROTO_PY}" ${GEN_PROTO_ARGS}
                       BYPRODUCTS "${GENERATED_PROTO}"
                       DEPENDS ${INFILE}
                       COMMENT "Running gen_proto.py on ${INFILE}"
                       )
    message("Generated: ${GENERATED_PROTO}")

    set(PROTOC_ARGS
        ${GENERATED_PROTO}
        -I
        ${CMAKE_CURRENT_BINARY_DIR}
        --cpp_out
        ${CMAKE_CURRENT_BINARY_DIR})
    if(ONNX_BUILD_PYTHON)
      list(APPEND PROTOC_ARGS --python_out)
      if(ONNX_GEN_PB_TYPE_STUBS)
        list(APPEND PROTOC_ARGS pyi_out:${CMAKE_CURRENT_BINARY_DIR})
      else()
        list(APPEND PROTOC_ARGS ${CMAKE_CURRENT_BINARY_DIR})
      endif()
    endif()
    list(APPEND GENERATED_FILE_TARGETS ${GENERATED_FILE_WE}_proto_file)
    add_custom_target(${GENERATED_FILE_WE}_src
        COMMAND "${ONNX_PROTOC_EXECUTABLE}" ${PROTOC_ARGS}
        BYPRODUCTS "${OUTPUT_PB_SRC}"
        DEPENDS ${GENERATED_FILE_TARGETS}
        COMMENT "Running C++ protocol buffer compiler on ${GENERATED_PROTO}")
  endforeach()

  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
endfunction()

relative_protobuf_generate_cpp(ONNX_PROTO_SRCS
                               onnx/onnx.in.proto
                               onnx/onnx-operators.in.proto
                               onnx/onnx-data.in.proto)


file(GLOB_RECURSE __tmp_srcs "${ONNX_ROOT}/onnx/*.cc")
file(GLOB_RECURSE onnx_gtests_src
    "${ONNX_ROOT}/onnx/test/cpp/*.cc"
    "${ONNX_ROOT}/onnx/backend/test/cpp/*.cc")
list(REMOVE_ITEM __tmp_srcs "${ONNX_ROOT}/onnx/cpp2py_export.cc")
list(REMOVE_ITEM __tmp_srcs ${onnx_gtests_src})
list(APPEND ONNX_SRCS ${__tmp_srcs})
list(APPEND ONNX_SRCS ${ONNX_PROTO_SRCS})

if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
  # whole-archive linker option not available on AIX.
  # So, create a object library
  add_library(onnx OBJECT ${ONNX_SRCS})
else()
  add_library(onnx ${ONNX_SRCS})
endif()
set_target_properties(onnx PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(onnx PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

set(LINKED_PROTOBUF_TARGET protobuf::libprotobuf)
if(ONNX_USE_LITE_PROTO)
  if(TARGET protobuf::libprotobuf-lite)
    set(LINKED_PROTOBUF_TARGET protobuf::libprotobuf-lite)
  endif()
endif()
target_link_libraries(onnx PUBLIC ${LINKED_PROTOBUF_TARGET})
foreach(ABSL_USED_TARGET IN LISTS protobuf_ABSL_USED_TARGETS)
  if(TARGET ${ABSL_USED_TARGET})
    target_link_libraries(onnx PRIVATE ${ABSL_USED_TARGET})
  endif()
endforeach()

add_onnx_global_defines(onnx)
target_include_directories(onnx PUBLIC
  $<BUILD_INTERFACE:${ONNX_ROOT}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)

if(ONNX_BUILD_PYTHON)
  # find system pybind11
  find_package(pybind11)
  if(NOT pybind11_FOUND)
    include(FetchContent)
    set(pybind11URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz)
    set(pybind11SHA1  8c7e3e8fec829ced31a495dec281153511f33c63)
    FetchContent_Declare(
      pybind11
      URL ${pybind11URL}
      URL_HASH SHA1=${pybind11SHA1}
    )
    FetchContent_MakeAvailable(pybind11)
  endif()

  pybind11_add_module(onnx_cpp2py_export MODULE "${ONNX_ROOT}/onnx/cpp2py_export.cc")

  if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
    # whole-archive linker option not available on AIX
    target_sources(onnx_cpp2py_export PRIVATE $<TARGET_OBJECTS:onnx>)
  else()
    target_link_libraries(onnx_cpp2py_export PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,onnx>)
  endif()
  # Prevent "undefined symbol: _ZNSt10filesystem7__cxx114path14_M_split_cmptsEv"
  # (std::filesystem::__cxx11::path::_M_split_cmpts()) on gcc 8
  if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(onnx_cpp2py_export PRIVATE "-lstdc++fs")
  endif()
endif()

if(MSVC)
  add_msvc_runtime_flag(onnx)
  if(TARGET onnx_cpp2py_export)
    add_msvc_runtime_flag(onnx_cpp2py_export)
  endif()
  if(ONNX_WERROR)
    target_compile_options(onnx PRIVATE "/WX")
  endif()
else()
  target_compile_options(onnx PRIVATE -Wall -Wextra)
  if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    target_compile_options(onnx PRIVATE "-Wno-stringop-overflow")
  endif()
  if(ONNX_WERROR)
    target_compile_options(onnx PRIVATE "-Werror")
  endif()
endif()
if(ONNX_USE_ASAN AND NOT MSVC)
  find_package(Sanitizer REQUIRED)
  if(TARGET Sanitizer::address)
    target_link_libraries(onnx PRIVATE Sanitizer::address)
    message(STATUS "Use ASAN")
  endif()
  if(TARGET Sanitizer::undefined)
    target_link_libraries(onnx PRIVATE Sanitizer::undefined)
    message(STATUS "Use UBSAN")
  endif()
endif()

install(DIRECTORY ${ONNX_ROOT}/onnx
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "backend/test/case" EXCLUDE
        PATTERN "backend/test/data" EXCLUDE)
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onnx
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.h")

configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/ONNXConfigVersion.cmake.in
  ${PROJECT_BINARY_DIR}/ONNXConfigVersion.cmake
  @ONLY)
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/ONNXConfig.cmake.in
  ${PROJECT_BINARY_DIR}/ONNXConfig.cmake
  @ONLY)
install(FILES
  ${PROJECT_BINARY_DIR}/ONNXConfigVersion.cmake
  ${PROJECT_BINARY_DIR}/ONNXConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ONNX
  COMPONENT dev)

if(ONNX_USE_UNITY_BUILD)
  # If ONNX_USE_UNITY_BUILD is set to ON, set ONNX target to use Unity builds.
  #  We set Unity build to use groups, it allows us to set some specific files to be compiled individually
  set_target_properties(onnx
    PROPERTIES
      UNITY_BUILD ON
      UNITY_BUILD_MODE GROUP
  )

  set(NEW_LIST __unity_src_files)
  list(APPEND __unity_src_files ${ONNX_SRCS})
  # These files have an issue with template explicit specialization after instantiation:
  #   We take them out of the unity group so they are compiled individually.
  list(REMOVE_ITEM __unity_src_files "${ONNX_ROOT}/onnx/defs/schema.cc")
  list(REMOVE_ITEM __unity_src_files "${ONNX_ROOT}/onnx/defs/tensor_proto_util.cc")
  set_source_files_properties(${__unity_src_files} PROPERTIES UNITY_GROUP "Unity_Group" )
  set_source_files_properties(${ONNX_PROTO_SRCS} PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

  # With unity build object file could get big, need this switch in MSVC.
  if(MSVC)
    target_compile_options(onnx PRIVATE /bigobj)
  endif()
endif()

if(ONNX_BUILD_TESTS)
  if(BUILD_SHARED_LIBS)
    message(FATAL_ERROR "Tests requires static build")
  endif()
  find_package(GTest)
  if(NOT GTest_FOUND)
    include(googletest)
  endif()
endif()

install(TARGETS
  onnx
  EXPORT ONNXTargets DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(EXPORT ONNXTargets
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ONNX"
  NAMESPACE ONNX::
)

if(ONNX_BUILD_TESTS)
  include(${ONNX_ROOT}/cmake/unittest.cmake)
endif()

include(cmake/summary.cmake)
onnx_print_configuration_summary()
