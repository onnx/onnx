# Minimum CMake required
cmake_minimum_required(VERSION 3.26)
# Project
project(onnx LANGUAGES CXX)

include(cmake/Utils.cmake)
# Honor visibility properties for all target types.
cmake_policy(SET CMP0063 NEW)
# Modules FindPython3 and FindPython use LOCATION for lookup strategy.
cmake_policy(SET CMP0094 NEW)

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


option(ONNX_BUILD_PYTHON "Build Python binaries" OFF)
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
  add_compile_options(
    /wd5287  # https://developercommunity.visualstudio.com/t/warning-C5287:-operands-are-different-e/10877942?sort=newest
    /Zc:lambda  # https://developercommunity.visualstudio.com/t/fatal--error-C1001:-Internal-compiler-er/10906076
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
  set(python_dev_component Development.Module Development.SABIModule)
endif()

if(CMAKE_CROSSCOMPILING)
  # When cross-compiling, the interpreter and the compiling/linking steps
  # must use a different package. See the discussion about this at
  # https://gitlab.kitware.com/cmake/cmake/-/issues/25145
  if(ONNX_BUILD_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS ${python_dev_component})
  endif()
  find_package(Python REQUIRED COMPONENTS Interpreter)
  set(ONNX_PYTHON_INTERPRETER Python::Interpreter)
else()
  find_package(Python3 REQUIRED COMPONENTS Interpreter ${python_dev_component})
  # Find Python for nanobind
  set(Python_EXECUTABLE ${Python3_EXECUTABLE})
  find_package(Python REQUIRED COMPONENTS Interpreter ${python_dev_component})
  set(ONNX_PYTHON_INTERPRETER Python3::Interpreter)
endif()

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
    set(ProtobufURL https://github.com/protocolbuffers/protobuf/releases/download/v31.1/protobuf-31.1.tar.gz)
    set(ProtobufSHA1 da10aaa3bf779735a8a9acde1256a47ce5d148be)
    FetchContent_Declare(
      Protobuf
      URL ${ProtobufURL}
      URL_HASH SHA1=${ProtobufSHA1}
    )
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build protobuf tests" FORCE)
    message(STATUS "Download and build Protobuf from ${ProtobufURL}")
    FetchContent_MakeAvailable(Protobuf)
    set(ONNX_PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
    set(Protobuf_VERSION "6.31.1")
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
  set(GENERATED_FILES)
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

    # Use add_custom_command to avoid re-generate of PROTO files
    add_custom_command(OUTPUT "${GENERATED_PROTO}"
        COMMAND ${ONNX_PYTHON_INTERPRETER} "${GEN_PROTO_PY}" ${GEN_PROTO_ARGS}
        DEPENDS ${INFILE}
        COMMENT "Running gen_proto.py on ${INFILE}")
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
    list(APPEND GENERATED_FILES "${GENERATED_PROTO}")
    add_custom_command(OUTPUT "${OUTPUT_PB_SRC}"
        COMMAND "${ONNX_PROTOC_EXECUTABLE}" ${PROTOC_ARGS}
        DEPENDS ${GENERATED_FILES}
        COMMENT "Running C++ protocol buffer compiler on ${GENERATED_PROTO}")
  endforeach()

  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
endfunction()

relative_protobuf_generate_cpp(ONNX_PROTO_SRCS
                               onnx/onnx.in.proto
                               onnx/onnx-operators.in.proto
                               onnx/onnx-data.in.proto)

add_library(onnx_proto_object OBJECT ${ONNX_PROTO_SRCS})

file(GLOB_RECURSE __tmp_srcs "${ONNX_ROOT}/onnx/*.h" "${ONNX_ROOT}/onnx/*.cc")
file(GLOB_RECURSE onnx_gtests_src "${ONNX_ROOT}/onnx/test/cpp/*.h"
    "${ONNX_ROOT}/onnx/test/cpp/*.cc"
    "${ONNX_ROOT}/onnx/backend/test/cpp/*.cc"
    "${ONNX_ROOT}/onnx/backend/test/cpp/*.h")
list(REMOVE_ITEM __tmp_srcs "${ONNX_ROOT}/onnx/cpp2py_export.cc")
list(REMOVE_ITEM __tmp_srcs ${onnx_gtests_src})
list(APPEND ONNX_SRCS ${__tmp_srcs})

# Hide all symbols we don't need
set_target_properties(onnx_proto_object PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(onnx_proto_object PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

set(LINKED_PROTOBUF_TARGET protobuf::libprotobuf)
if(ONNX_USE_LITE_PROTO)
  if(TARGET protobuf::libprotobuf-lite)
    set(LINKED_PROTOBUF_TARGET protobuf::libprotobuf-lite)
  endif()
endif()
add_onnx_global_defines(onnx_proto_object)
target_include_directories(onnx_proto_object PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
if(MSVC)
  # For disabling Protobuf related warnings
  set(protobuf_warnings
    /wd4146 # unary minus operator applied to unsigned type,
            # result still unsigned
    /wd4244 # 'argument': conversion from 'google::
            # protobuf::uint64' to 'int', possible
            # loss of data
    /wd4267 # Conversion from 'size_t' to 'int',
            # possible loss of data
    /wd4141 # 'inline': used more than once
  )
endif()
add_library(onnx_proto)
target_link_libraries(onnx_proto PUBLIC $<BUILD_INTERFACE:onnx_proto_object>)

# onnx_object and onnx_proto_object are collections of C++ object files.
# They are introduced to help onnx_cpp2py_export bypass library boundaries and use ONNX private symbols.
add_library(onnx_object OBJECT ${ONNX_SRCS})
add_dependencies(onnx_object onnx_proto_object)
target_include_directories(onnx_object PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
set_target_properties(onnx_object PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(onnx_object PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
target_include_directories(onnx_object PUBLIC $<BUILD_INTERFACE:${ONNX_ROOT}>)
add_onnx_global_defines(onnx_object)
target_link_libraries(onnx_proto_object PUBLIC ${LINKED_PROTOBUF_TARGET})
target_link_libraries(onnx_object PUBLIC ${LINKED_PROTOBUF_TARGET})
foreach(ABSL_USED_TARGET IN LISTS protobuf_ABSL_USED_TARGETS)
  if(TARGET ${ABSL_USED_TARGET})
    target_link_libraries(onnx_proto_object PUBLIC ${ABSL_USED_TARGET})
    target_link_libraries(onnx_object PUBLIC ${ABSL_USED_TARGET})
  endif()
endforeach()
add_library(onnx)
target_link_libraries(onnx PUBLIC $<BUILD_INTERFACE:onnx_object>)
target_link_libraries(onnx PUBLIC onnx_proto)
target_compile_options(onnx_object PUBLIC ${protobuf_warnings})
target_compile_options(onnx_proto_object PUBLIC ${protobuf_warnings})

if(ONNX_BUILD_PYTHON)
  # find system nanobind
  find_package(nanobind)
  if(NOT nanobind_FOUND)
    include(FetchContent)

    FetchContent_Declare(
      nanobind
      GIT_REPOSITORY https://github.com/wjakob/nanobind.git
      GIT_TAG v2.8.0
      GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nanobind)
  endif()

  # Configure nanobind: https://nanobind.readthedocs.io/en/latest/api_cmake.html
  nanobind_add_module(
    onnx_cpp2py_export
    NB_STATIC NB_DOMAIN onnx STABLE_ABI FREE_THREADED LTO
    "${ONNX_ROOT}/onnx/cpp2py_export.cc")

  target_link_libraries(onnx_cpp2py_export PRIVATE $<BUILD_INTERFACE:onnx_object> $<BUILD_INTERFACE:onnx_proto_object>)
  # Prevent "undefined symbol: _ZNSt10filesystem7__cxx114path14_M_split_cmptsEv"
  # (std::filesystem::__cxx11::path::_M_split_cmpts()) on gcc 8
  if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(onnx_cpp2py_export PRIVATE "-lstdc++fs")
  endif()
endif()

if(MSVC)
  add_msvc_runtime_flag(onnx_proto_object)
  add_msvc_runtime_flag(onnx_object)
  if(TARGET onnx_cpp2py_export)
    add_msvc_runtime_flag(onnx_cpp2py_export)
  endif()
  if(ONNX_WERROR)
    target_compile_options(onnx_object PRIVATE "/WX")
  endif()
else()
  target_compile_options(onnx_object PRIVATE -Wall -Wextra)
  if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
    target_compile_options(onnx_object PRIVATE "-Wno-stringop-overflow")
  endif()
  if(ONNX_WERROR)
    target_compile_options(onnx_object PRIVATE "-Werror")
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
  #   We take them out of the unity group so that they are compiled individually.
  list(REMOVE_ITEM __unity_src_files "${ONNX_ROOT}/onnx/defs/schema.cc")
  list(REMOVE_ITEM __unity_src_files "${ONNX_ROOT}/onnx/defs/tensor_proto_util.cc")
  set_source_files_properties(${__unity_src_files} PROPERTIES UNITY_GROUP "Unity_Group" )

  # With unity build object file could get big, need this switch in MSVC.
  if(MSVC)
    target_compile_options(onnx PRIVATE /bigobj)
  endif()
# should be enabled for onnx_proto when protobuf can support Unity builds
endif()

if(ONNX_BUILD_TESTS)
  find_package(GTest)
  if(NOT GTest_FOUND)
    include(googletest)
  endif()
endif()

install(TARGETS
  onnx onnx_proto
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
