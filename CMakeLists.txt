# Minimum CMake required
cmake_minimum_required(VERSION 3.1)
include(cmake/Utils.cmake)
# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(
    CMAKE_BUILD_TYPE "Release"
    CACHE
      STRING
      "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage."
    FORCE)
endif()
cmake_policy(SET CMP0063 NEW)

# Project
project(onnx C CXX)
option(ONNX_BUILD_BENCHMARKS "Build ONNX micro-benchmarks" OFF)

option(BUILD_ONNX_PYTHON "Build Python binaries" OFF)
option(ONNX_GEN_PB_TYPE_STUBS "Generate protobuf python type stubs" ON)
option(ONNX_WERROR "Build with Werror" OFF)
option(ONNX_COVERAGE "Build with coverage instrumentation" OFF)
option(ONNX_BUILD_TESTS "Build ONNX C++ APIs Tests" OFF)

# Set C++11 as standard for the whole project
if(NOT MSVC)
  set(CMAKE_CXX_STANDARD 11)
endif(NOT MSVC)

set(ONNX_ROOT ${PROJECT_SOURCE_DIR})

set(CMAKE_MODULE_PATH "")
list(APPEND CMAKE_MODULE_PATH ${ONNX_ROOT}/cmake/Modules)

if(NOT MSVC)
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
  if(ONNX_COVERAGE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  endif()
endif()

if(ONNX_BUILD_TESTS)
  list(APPEND CMAKE_MODULE_PATH ${ONNX_ROOT}/cmake/external)
  include(googletest)
endif()

if(TARGET protobuf::libprotobuf)
  # Sometimes we need to use protoc compiled for host architecture while linking
  # libprotobuf against target architecture. See https://github.com/caffe2/caffe
  # 2/blob/96f35ad75480b25c1a23d6e9e97bccae9f7a7f9c/cmake/ProtoBuf.cmake#L92-L99
  if(EXISTS "${ONNX_CUSTOM_PROTOC_EXECUTABLE}")
    message(STATUS "Using custom protoc executable")
    set(ONNX_PROTOC_EXECUTABLE ${ONNX_CUSTOM_PROTOC_EXECUTABLE})
  else()
    set(ONNX_PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
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
    find_library(Protobuf_LIBRARY
                 NAMES protobuf
                 PATHS ${_PROTOBUF_INSTALL_PREFIX}/lib
                 NO_DEFAULT_PATH)
    find_library(Protobuf_PROTOC_LIBRARY
                 NAMES protoc
                 PATHS ${_PROTOBUF_INSTALL_PREFIX}/lib
                 NO_DEFAULT_PATH)
    find_library(Protobuf_LITE_LIBRARY
                 NAMES protobuf-lite
                 PATHS ${_PROTOBUF_INSTALL_PREFIX}/lib
                 NO_DEFAULT_PATH)
    find_path(Protobuf_INCLUDE_DIR google/protobuf/service.h
              PATHS ${_PROTOBUF_INSTALL_PREFIX}/include
              NO_DEFAULT_PATH)
    find_package(Protobuf REQUIRED)
  endif()
endif()

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(ONNX_NAMESPACE)
  set(MY_ONNX_NAMESPACE "-DONNX_NAMESPACE=${ONNX_NAMESPACE}")
else()
  set(ONNX_NAMESPACE "onnx")
  set(MY_ONNX_NAMESPACE "-DONNX_NAMESPACE=onnx")
endif()
add_definitions(${MY_ONNX_NAMESPACE})

if(ONNX_ML)
  add_definitions("-DONNX_ML=1")
endif()

# function(RELATIVE_PROTOBUF_GENERATE_CPP SRCS HDRS ROOT_DIR) from https://githu
# b.com/tensorflow/tensorflow/blob/d2c3b873c6f8ff999a2e4ee707a84ff00d9c15a5/tens
# orflow/contrib/cmake/tf_core_framework.cmake to solve the problem that
# customized dir can't be specified when calling PROTOBUF_GENERATE_CPP.
function(RELATIVE_PROTOBUF_GENERATE_CPP NAME SRCS HDRS ROOT_DIR DEPEND)
  if(NOT ARGN)
    message(
      SEND_ERROR
        "Error: RELATIVE_PROTOBUF_GENERATE_CPP() called without any proto files"
      )
    return()
  endif()

  if(MSVC AND BUILD_SHARED_LIBS)
    set(ONNX_DLLEXPORT_STR "dllexport_decl=ONNX_API:")
  else()
    set(ONNX_DLLEXPORT_STR "")
  endif()

  set(${SRCS})
  set(${HDRS})

  set(GEN_PROTO_PY ${ROOT_DIR}/onnx/gen_proto.py)
  foreach(INFILE ${ARGN})
    set(ABS_FILE ${ROOT_DIR}/${INFILE})
    get_filename_component(FILE_DIR ${ABS_FILE} DIRECTORY)
    get_filename_component(FILE_WE ${INFILE} NAME_WE)
    if(ONNX_ML)
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
    file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FILE_DIR})
    set(OUTPUT_PROTO_DIR "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}")

    set(OUTPUT_PB_HEADER "${OUTPUT_PROTO_DIR}/${GENERATED_FILE_WE}.pb.h")
    set(OUTPUT_PB_SRC "${OUTPUT_PROTO_DIR}/${GENERATED_FILE_WE}.pb.cc")
    set(GENERATED_PROTO "${OUTPUT_PROTO_DIR}/${GENERATED_FILE_WE}.proto")
    if(NOT (ONNX_NAMESPACE STREQUAL "onnx"))
      # We need this dummy header generated by gen_proto.py when ONNX_NAMESPACE
      # is not onnx
      list(APPEND ${HDRS} "${OUTPUT_PROTO_DIR}/${GENERATED_FILE_WE}.pb.h")
    endif()
    list(APPEND ${SRCS} "${OUTPUT_PB_SRC}")
    list(APPEND ${HDRS} "${OUTPUT_PB_HEADER}")

    if(NOT EXISTS "${OUTPUT_PROTO_DIR}")
      file(MAKE_DIRECTORY "${OUTPUT_PROTO_DIR}")
    endif()

    if("${PYTHON_EXECUTABLE}" STREQUAL "")
      set(_python_exe "python")
    else()
      set(_python_exe "${PYTHON_EXECUTABLE}")
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
    add_custom_command(OUTPUT "${GENERATED_PROTO}"
                       COMMAND "${_python_exe}" "${GEN_PROTO_PY}"
                               ARGS ${GEN_PROTO_ARGS}
                       DEPENDS ${INFILE}
                       COMMENT "Running gen_proto.py on ${INFILE}"
                       VERBATIM)

    set(PROTOC_ARGS
        ${GENERATED_PROTO}
        -I
        ${CMAKE_CURRENT_BINARY_DIR}
        --cpp_out
        ${ONNX_DLLEXPORT_STR}${CMAKE_CURRENT_BINARY_DIR})
    if(BUILD_ONNX_PYTHON)
      list(APPEND PROTOC_ARGS --python_out
                  ${ONNX_DLLEXPORT_STR}${CMAKE_CURRENT_BINARY_DIR})
      if(ONNX_GEN_PB_TYPE_STUBS)
        # Haven't figured out how to generate mypy stubs on Windows yet
        if(NOT WIN32)
          # If onnx was packaged to pypi from Windows, protoc-gen-mypy.py is
          # missing the +x flag. Add it.
          execute_process(COMMAND chmod +x ${ROOT_DIR}/tools/protoc-gen-mypy.py)
          set(PROTOC_MYPY_PLUGIN_FILE ${ROOT_DIR}/tools/protoc-gen-mypy.py)
        else(NOT WIN32)
          set(PROTOC_MYPY_PLUGIN_FILE ${ROOT_DIR}/tools/protoc-gen-mypy.bat)
        endif()
        list(APPEND PROTOC_ARGS
                    --plugin
                    protoc-gen-mypy=${PROTOC_MYPY_PLUGIN_FILE}
                    --mypy_out
                    ${ONNX_DLLEXPORT_STR}${CMAKE_CURRENT_BINARY_DIR})
      endif()
    endif()
    if(NOT ONNX_PROTOC_EXECUTABLE)
      message(FATAL_ERROR "Protobuf compiler not found")
    endif()
    if(ONNX_PROTO_POST_BUILD_SCRIPT)
      add_custom_command(
        OUTPUT "${OUTPUT_PB_SRC}" "${OUTPUT_PB_HEADER}"
        COMMAND ${ONNX_PROTOC_EXECUTABLE} ARGS ${PROTOC_ARGS}
        COMMAND "${CMAKE_COMMAND}" -DFILENAME=${OUTPUT_PB_HEADER}
                -DNAMESPACES=${ONNX_NAMESPACE} -P
                ${ONNX_PROTO_POST_BUILD_SCRIPT}
        COMMAND "${CMAKE_COMMAND}" -DFILENAME=${OUTPUT_PB_SRC}
                -DNAMESPACES=${ONNX_NAMESPACE} -P
                ${ONNX_PROTO_POST_BUILD_SCRIPT}
        DEPENDS ${GENERATED_PROTO} ${DEPEND}
        COMMENT "Running C++ protocol buffer compiler on ${GENERATED_PROTO}"
        VERBATIM)
    else()
      add_custom_command(
        OUTPUT "${OUTPUT_PB_SRC}" "${OUTPUT_PB_HEADER}"
        COMMAND ${ONNX_PROTOC_EXECUTABLE} ARGS ${PROTOC_ARGS}
        DEPENDS ${GENERATED_PROTO} ${DEPEND}
        COMMENT "Running C++ protocol buffer compiler on ${GENERATED_PROTO}"
        VERBATIM)
    endif()
    add_custom_target(${NAME} DEPENDS ${OUTPUT_PB_SRC} ${OUTPUT_PB_HEADER})
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

relative_protobuf_generate_cpp(gen_onnx_proto
                               PROTO_SRCS
                               PROTO_HDRS
                               ${ONNX_ROOT}
                               ""
                               onnx/onnx.in.proto)
relative_protobuf_generate_cpp(gen_onnx_operators_proto
                               PROTO_SRCS2
                               PROTO_HDRS2
                               ${ONNX_ROOT}
                               gen_onnx_proto
                               onnx/onnx-operators.in.proto)
list(APPEND PROTO_SRCS ${PROTO_SRCS2})
list(APPEND PROTO_HDRS ${PROTO_HDRS2})

file(GLOB_RECURSE onnx_src "${ONNX_ROOT}/onnx/*.h" "${ONNX_ROOT}/onnx/*.cc")
file(GLOB_RECURSE onnx_gtests_src "${ONNX_ROOT}/onnx/test/cpp/*.h"
                  "${ONNX_ROOT}/onnx/test/cpp/*.cc")
list(REMOVE_ITEM onnx_src "${ONNX_ROOT}/onnx/cpp2py_export.cc")
list(REMOVE_ITEM onnx_src ${onnx_gtests_src})

add_library(onnx_proto ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(onnx_proto
                           PUBLIC "${CMAKE_CURRENT_BINARY_DIR}"
                                  "${PROTOBUF_INCLUDE_DIRS}")
if(TARGET protobuf::libprotobuf)
  target_link_libraries(onnx_proto PUBLIC protobuf::libprotobuf)
else()
  target_link_libraries(onnx_proto PUBLIC ${PROTOBUF_LIBRARIES})
endif()

add_library(onnx ${onnx_src})
target_include_directories(onnx
                           PUBLIC ${ONNX_ROOT} "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(onnx PUBLIC onnx_proto)

if(BUILD_ONNX_PYTHON)
  if("${PY_EXT_SUFFIX}" STREQUAL "")
    if(MSVC)
      set(PY_EXT_SUFFIX ".pyd")
    else()
      set(PY_EXT_SUFFIX ".so")
    endif()
  endif()

  add_library(onnx_cpp2py_export MODULE "${ONNX_ROOT}/onnx/cpp2py_export.cc")
  set_target_properties(onnx_cpp2py_export PROPERTIES PREFIX "")
  set_target_properties(onnx_cpp2py_export
                        PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
  set_target_properties(onnx_cpp2py_export PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
  set_target_properties(onnx_cpp2py_export
                        PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_include_directories(onnx_cpp2py_export
                             PRIVATE "${ONNX_ROOT}"
                                     "${CMAKE_CURRENT_BINARY_DIR}"
                                     "${PROTOBUF_INCLUDE_DIRS}"
                                     "${PYTHON_INCLUDE_DIR}")

  # pybind11 is a header only lib
  find_package(pybind11)
  if(pybind11_FOUND)
    target_include_directories(onnx_cpp2py_export
                               PRIVATE ${pybind11_INCLUDE_DIRS})
  else()
    target_include_directories(
      onnx_cpp2py_export
      PRIVATE ${ONNX_ROOT}/third_party/pybind11/include)
  endif()

  if(APPLE)
    set_target_properties(onnx_cpp2py_export
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
    target_link_libraries(onnx_cpp2py_export
                          PRIVATE -Wl,-force_load,$<TARGET_FILE:onnx>)
  elseif(MSVC)
    # In MSVC, we will add whole archive in default
    target_link_libraries(onnx_cpp2py_export
                          PRIVATE -WHOLEARCHIVE:$<TARGET_FILE:onnx>)
  else()
    # Assume everything else is like gcc
    target_link_libraries(onnx_cpp2py_export
                          PRIVATE "-Wl,--whole-archive" $<TARGET_FILE:onnx>
                                  "-Wl,--no-whole-archive")
  endif()

  target_link_libraries(onnx_cpp2py_export PRIVATE onnx)

  if(MSVC)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(MP_FLAG "-Xclang" "-fopenmp")
      set(EXTRA_FLAGS
          "-Wno-implicit-function-declaration"
          "-Wno-undefined-inline"
          "-Wno-incompatible-pointer-types"
          "-Wno-dllexport-explicit-instantiation-decl"
          "-Wno-microsoft-unqualified-friend"
          "-Wno-absolute-value"
          "-Wno-unused-variable"
          "-Wno-writable-strings"
          "-Qunused-arguments")
    else()
      set(MP_FLAG "/MP")
      set(EXTRA_FLAGS "")
    endif()
    find_package(PythonInterp ${PY_VERSION} REQUIRED)
    find_package(PythonLibs ${PY_VERSION} REQUIRED)
    target_link_libraries(onnx_cpp2py_export PRIVATE ${PYTHON_LIBRARIES})
    target_compile_options(onnx_cpp2py_export
                           PRIVATE ${MP_FLAG}
                                   /WX
                                   /wd4800 # disable warning type' : forcing
                                           # value to bool 'true' or 'false'
                                           # (performance warning)
                                   /wd4503 # identifier' : decorated name length
                                           # exceeded, name was truncated
                                   /wd4146 # unary minus operator applied to
                                           # unsigned type, result still
                                           # unsigned from include\google\protob
                                           # uf\wire_format_lite.h
                                   ${EXTRA_FLAGS})
    target_compile_options(onnx_cpp2py_export PRIVATE /MT)
  endif()
endif()

if(ONNX_BUILD_BENCHMARKS)
  if(NOT TARGET benchmark)
    # We will not need to test benchmark lib itself.
    set(BENCHMARK_ENABLE_TESTING OFF
        CACHE BOOL "Disable benchmark testing as we don't need it.")
    # We will not need to install benchmark since we link it statically.
    set(BENCHMARK_ENABLE_INSTALL OFF
        CACHE BOOL
              "Disable benchmark install to avoid overwriting vendor install.")
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/benchmark)
  endif()

  add_executable(protobuf-bench tools/protobuf-bench.cc)
  target_include_directories(protobuf-bench
                             PUBLIC "${ONNX_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}"
                                    "${PROTOBUF_INCLUDE_DIRS}")
  target_link_libraries(protobuf-bench onnx_proto benchmark)
endif()

# Export include directories
set(ONNX_INCLUDE_DIRS "${ONNX_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}")
get_directory_property(hasParent PARENT_DIRECTORY)
if(hasParent)
  set(ONNX_INCLUDE_DIRS ${ONNX_INCLUDE_DIRS} PARENT_SCOPE)
endif()

if(MSVC)
  target_compile_options(onnx_proto
                         PRIVATE ${MP_FLAG}
                                 /WX
                                 /wd4800 # disable warning type' : forcing value
                                         # to bool 'true' or 'false'
                                         # (performance warning)
                                 /wd4503 # identifier' : decorated name length
                                         # exceeded, name was truncated
                                 /wd4146 # unary minus operator applied to
                                         # unsigned type, result still unsigned:
                                         # include\google\protobuf\wire_format_l
                                         # ite.h
                                 ${EXTRA_FLAGS})
  target_compile_options(onnx
                         PRIVATE ${MP_FLAG}
                                 /WX
                                 /wd4800 # disable warning type' : forcing value
                                         # to bool 'true' or 'false'
                                         # (performance warning)
                                 /wd4503 # identifier' : decorated name length
                                         # exceeded, name was truncated
                                 /wd4146 # unary minus operator applied to
                                         # unsigned type, result still unsigned
                                 ${EXTRA_FLAGS})
  add_msvc_runtime_flag(onnx_proto)
  add_msvc_runtime_flag(onnx)
  set(onnx_static_library_flags
      -IGNORE:4221 # LNK4221: This object file does not define any previously
                   # undefined public symbols, so it will not be used by any
                   # link operation that consumes this library
      )
  set_target_properties(onnx
                        PROPERTIES STATIC_LIBRARY_FLAGS
                                   "${onnx_static_library_flags}")
elseif(APPLE)

else()
  if(${ONNX_WERROR})
    target_compile_options(onnx PRIVATE -Werror=sign-compare -Werror=conversion)
  endif()
endif()

if(APPLE)
  set_target_properties(onnx PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

# ---[ ONNX Interface for Framework Integratin (ONNXIFI)
add_library(onnxifi INTERFACE)
target_include_directories(onnxifi INTERFACE onnx)

# ---[ ONNXIFI loader
add_library(onnxifi_loader STATIC onnx/onnxifi_loader.c)
# Users of ONNX backend API would compile it with their toolchain, so it is
# implemented in standard C89 for maximum compatibility
set_target_properties(onnxifi_loader
                      PROPERTIES C_STANDARD
                                 90
                                 C_EXTENSIONS
                                 NO)
target_link_libraries(onnxifi_loader PUBLIC onnxifi ${CMAKE_DL_LIBS})
if(MSVC)
  add_msvc_runtime_flag(onnxifi_loader)
endif()
target_include_directories(onnxifi_loader PUBLIC onnx)
target_include_directories(onnxifi_loader PUBLIC ${ONNX_ROOT})

# ---[ ONNXIFI dummy backend
add_library(onnxifi_dummy SHARED onnx/onnxifi_dummy.c)
target_link_libraries(onnxifi_dummy PUBLIC onnxifi ${CMAKE_DL_LIBS})
target_include_directories(onnxifi_dummy PUBLIC onnx)
target_include_directories(onnxifi_dummy PUBLIC ${ONNX_ROOT})
target_compile_definitions(onnxifi_dummy PRIVATE ONNXIFI_BUILD_LIBRARY=TRUE)
if(MSVC)
  add_msvc_runtime_flag(onnxifi_dummy)
endif()
install(TARGETS onnx onnx_proto DESTINATION lib)
install(DIRECTORY ${ONNX_ROOT}/onnx
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/onnx
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h")
install(TARGETS onnx onnx_proto onnxifi_dummy DESTINATION lib)
if(ONNX_BUILD_TESTS)
  include(${ONNX_ROOT}/cmake/unittest.cmake)
endif()
