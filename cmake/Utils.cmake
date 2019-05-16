function(add_onnx_global_defines target)
  target_compile_definitions(${target} PUBLIC "ONNX_NAMESPACE=${ONNX_NAMESPACE}")

  if(ONNX_ML)
    target_compile_definitions(${target} PUBLIC "ONNX_ML=1")
  endif()

  if(ONNX_USE_LITE_PROTO)
    target_compile_definitions(${target} PUBLIC "ONNX_USE_LITE_PROTO=1")
  endif()
endfunction()
