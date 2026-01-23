# SPDX-License-Identifier: Apache-2.0
#
# Compiler hardening flags based on OpenSSF guidelines:
# https://best.openssf.org/Compiler-Hardening-Guides/Compiler-Options-Hardening-Guide-for-C-and-C++.html
function(add_onnx_hardening_flags target)
  if(NOT ONNX_HARDENING)
    return()
  endif()

  if(MSVC)
    # MSVC hardening flags
    target_compile_options(${target} PRIVATE
      /GS           # Buffer security check
      /DYNAMICBASE  # ASLR
      /NXCOMPAT     # Data Execution Prevention
      /guard:cf     # Control Flow Guard
    )
    target_link_options(${target} PRIVATE
      /DYNAMICBASE
      /NXCOMPAT
      /GUARD:CF
    )
  else()
    # GCC/Clang hardening compile flags
    target_compile_options(${target} PRIVATE
      -Wformat
      -Wformat=2
      -Wimplicit-fallthrough
      -Werror=format-security
      -fstack-protector-strong
    )

    # _FORTIFY_SOURCE requires optimization and conflicts with sanitizers
    if(NOT ONNX_USE_ASAN)
      target_compile_options(${target} PRIVATE
        -U_FORTIFY_SOURCE
        -D_FORTIFY_SOURCE=3
      )
    endif()

    # C++ standard library assertions
    target_compile_definitions(${target} PRIVATE _GLIBCXX_ASSERTIONS)

    # Stack clash protection (not supported on all platforms)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-fstack-clash-protection COMPILER_SUPPORTS_STACK_CLASH)
    if(COMPILER_SUPPORTS_STACK_CLASH)
      target_compile_options(${target} PRIVATE -fstack-clash-protection)
    endif()

    # Control-flow protection for x86_64
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
      check_cxx_compiler_flag(-fcf-protection=full COMPILER_SUPPORTS_CF_PROTECTION)
      if(COMPILER_SUPPORTS_CF_PROTECTION)
        target_compile_options(${target} PRIVATE -fcf-protection=full)
      endif()
    endif()

    # Branch protection for AArch64
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
      check_cxx_compiler_flag(-mbranch-protection=standard COMPILER_SUPPORTS_BRANCH_PROTECTION)
      if(COMPILER_SUPPORTS_BRANCH_PROTECTION)
        target_compile_options(${target} PRIVATE -mbranch-protection=standard)
      endif()
    endif()

    # Linker hardening flags (Linux only, not macOS)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_link_options(${target} PRIVATE
        -Wl,-z,noexecstack
        -Wl,-z,relro
        -Wl,-z,now
      )
    endif()
  endif()
endfunction()

# Add MSVC RunTime Flag
function(add_msvc_runtime_flag lib)
  if(ONNX_USE_MSVC_STATIC_RUNTIME)
    target_compile_options(${lib} PRIVATE $<$<NOT:$<CONFIG:Debug>>:/MT>
                                          $<$<CONFIG:Debug>:/MTd>)
  else()
    target_compile_options(${lib} PRIVATE $<$<NOT:$<CONFIG:Debug>>:/MD>
                                          $<$<CONFIG:Debug>:/MDd>)
  endif()
endfunction()

function(add_onnx_global_defines target)
  target_compile_definitions(${target}
                             PUBLIC "ONNX_NAMESPACE=${ONNX_NAMESPACE}")

  if(ONNX_ML)
    target_compile_definitions(${target} PUBLIC "ONNX_ML=1")
  endif()

  if(ONNX_USE_LITE_PROTO)
    target_compile_definitions(${target} PUBLIC "ONNX_USE_LITE_PROTO=1")
  endif()

  if(ONNX_DISABLE_STATIC_REGISTRATION)
    target_compile_definitions(${target}
                               PUBLIC "__ONNX_DISABLE_STATIC_REGISTRATION")
  endif()
endfunction()

function(add_onnx_compile_options target)
  if(MSVC)
    # For disabling Protobuf related warnings
    set(protobuf_warnings
        /wd4146 # unary minus operator applied to unsigned type, result still
                # unsigned
        /wd4244 # 'argument': conversion from 'google::protobuf::uint64' to
                # 'int', possible loss of data
        /wd4267 # Conversion from 'size_t' to 'int', possible loss of data
        /wd4141 # 'inline': used more than once
        /wd4047 # '=': 'uintptr_t' differs in levels of indirection from 'void *'
    )
    add_msvc_runtime_flag(${target})
    target_compile_options(${target} PUBLIC ${protobuf_warnings})
    if(ONNX_WERROR)
      target_compile_options(${target} PRIVATE "/WX")
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra)
    if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION
                                    VERSION_GREATER_EQUAL 13)
      target_compile_options(${target} PRIVATE "-Wno-stringop-overflow")
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
      target_compile_options(${target} PRIVATE "-Wno-shorten-64-to-32")
    endif()
    if(ONNX_WERROR)
      target_compile_options(${target} PRIVATE "-Werror")
    endif()
  endif()
  target_include_directories(
    ${target}
    PUBLIC $<BUILD_INTERFACE:${ONNX_ROOT}> $<INSTALL_INTERFACE:include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
  target_link_libraries(${target} PUBLIC ${LINKED_PROTOBUF_TARGET})
  foreach(ABSL_USED_TARGET IN LISTS protobuf_ABSL_USED_TARGETS)
    if(TARGET ${ABSL_USED_TARGET})
      target_link_libraries(${target} PUBLIC ${ABSL_USED_TARGET})
    endif()
  endforeach()
  # Prevent "undefined symbol: _ZNSt10filesystem7__cxx114path14_M_split_cmptsEv"
  # (std::filesystem::__cxx11::path::_M_split_cmpts()) on gcc 8
  if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(${target} PRIVATE "-lstdc++fs")
  endif()

  # Apply hardening flags if enabled
  add_onnx_hardening_flags(${target})
endfunction()
