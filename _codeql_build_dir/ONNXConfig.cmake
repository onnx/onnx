# - Config file for the ONNX package
# It defines ONNX targets for other cmake libraries to use.

# library version information
set(ONNX_VERSION "1.21.0")

if(NOT OFF)
  find_package(Protobuf REQUIRED CONFIG)
endif()

# import targets
include ("${CMAKE_CURRENT_LIST_DIR}/ONNXTargets.cmake")
