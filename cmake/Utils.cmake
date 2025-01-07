# SPDX-License-Identifier: Apache-2.0
#
# Add MSVC RunTime Flag
function(add_msvc_runtime_flag lib)
  if(ONNX_USE_MSVC_STATIC_RUNTIME)
    target_compile_options(${lib} PRIVATE $<$<NOT:$<CONFIG:Debug>>:/MT> $<$<CONFIG:Debug>:/MTd>)    
  else()
    target_compile_options(${lib} PRIVATE $<$<NOT:$<CONFIG:Debug>>:/MD> $<$<CONFIG:Debug>:/MDd>)
  endif()
endfunction()

function(add_onnx_global_defines target)
  target_compile_definitions(${target} PUBLIC "ONNX_NAMESPACE=${ONNX_NAMESPACE}")

  if(ONNX_ML)
    target_compile_definitions(${target} PUBLIC "ONNX_ML=1")
  endif()

  if(ONNX_USE_LITE_PROTO)
    target_compile_definitions(${target} PUBLIC "ONNX_USE_LITE_PROTO=1")
  endif()

  if(ONNX_DISABLE_STATIC_REGISTRATION)
    target_compile_definitions(${target} PUBLIC "__ONNX_DISABLE_STATIC_REGISTRATION")
  endif()  
endfunction()

function(add_whole_archive_flag lib output_var)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
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
