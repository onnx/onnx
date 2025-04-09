# CMake file to replace the string contents
# Copied and modified from https://github.com/pytorch/pytorch/blob/main/cmake/ProtoBufPatch.cmake
# Usage example:
#   cmake -DFILENAME=CMakeLists.txt -P ProtoBufPatch.cmake

file(READ ${FILENAME} content)

string(
  REPLACE
  "set(ABSL_MSVC_STATIC_RUNTIME ON)"
  "set(ABSL_MSVC_STATIC_RUNTIME ${protobuf_MSVC_STATIC_RUNTIME})"
  content
  "${content}")

file(WRITE ${FILENAME} "${content}")
