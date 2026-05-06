# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Pyodide/Emscripten abseil linking configuration.
# Included from CMakeLists.txt when ONNX_PYODIDE_BUILD is ON.
#
# pyodide-build uses pywasmcross compiler wrappers that do not set
# CMAKE_SYSTEM_NAME=Emscripten, so standard cmake platform detection does not
# apply.  cmake imported absl targets carry transitive deps on
# absl::examine_stack, which defines HaveOffsetConverter as an EM_JS function.
# EM_JS functions cannot be linked into wasm side-modules and cause a hard
# linker error ("undefined exported symbol: ___em_js__HaveOffsetConverter").
#
# Instead of cmake targets, glob the abseil/utf8_range static libraries from
# the pre-built wasm install prefix and exclude the debugging sub-libraries
# that embed EM_JS functions.

if(NOT DEFINED Protobuf_LITE_LIBRARY OR NOT EXISTS "${Protobuf_LITE_LIBRARY}")
  message(FATAL_ERROR
    "ONNX_PYODIDE_BUILD requires Protobuf_LITE_LIBRARY to point to the "
    "pre-built wasm protobuf-lite static library")
endif()

get_filename_component(_pb_lib_dir "${Protobuf_LITE_LIBRARY}" DIRECTORY)

file(GLOB _absl_pyodide_libs LIST_DIRECTORIES false
  "${_pb_lib_dir}/libabsl_*.a"
  "${_pb_lib_dir}/libutf8_range.a"
  "${_pb_lib_dir}/libutf8_validity.a"
)

# Exclude debugging sub-libraries whose object files contain EM_JS functions
# (e.g. HaveOffsetConverter in examine_stack) — the Emscripten linker rejects
# EM_JS in wasm side-modules with a hard error.
list(FILTER _absl_pyodide_libs EXCLUDE REGEX
  "libabsl_(failure_signal_handler|symbolize|stacktrace|examine_stack|debugging_internal|demangle_internal|leak_check)\\.a$"
)

if(NOT _absl_pyodide_libs)
  message(FATAL_ERROR
    "No abseil static libraries found in '${_pb_lib_dir}' — "
    "ensure protobuf-wasm was built with ABSL_ENABLE_INSTALL=ON and "
    "cmake --install was run before this build")
endif()

message(STATUS "Pyodide: linking safe abseil/utf8 static libs from ${_pb_lib_dir}")
set(protobuf_ABSL_USED_TARGETS ${_absl_pyodide_libs})
