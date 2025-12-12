# SPDX-License-Identifier: Apache-2.0
#
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
endfunction()
