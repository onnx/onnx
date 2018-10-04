#
# Add MSVC RunTime Flag
function(add_msvc_runtime_flag lib)
  if(${ONNX_USE_MSVC_STATIC_RUNTIME})
    if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
      target_compile_options(${lib} PRIVATE /MTd)
    else()
      target_compile_options(${lib} PRIVATE /MT)
    endif()
  else()
    if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
      target_compile_options(${lib} PRIVATE /MDd)
    else()
      target_compile_options(${lib} PRIVATE /MD)
    endif()
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

  if (BUILD_SHARED_LIBS)
    target_compile_definitions(${target} PRIVATE "ONNX_BUILD_SHARED_LIBS=1")
  endif()

  if (ONNX_BUILD_MAIN_LIB)
    target_compile_definitions(${target} PRIVATE "ONNX_BUILD_MAIN_LIB=1")
  endif()
endfunction()
