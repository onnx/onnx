# User options
include("${CMAKE_CURRENT_LIST_DIR}/protobuf-options.cmake")

# Depend packages
if(NOT ZLIB_FOUND)
  find_package(ZLIB)
endif()
if(NOT TARGET absl::strings)
  find_package(absl CONFIG)
endif()
if(NOT TARGET utf8_range)
  find_package(utf8_range CONFIG)
endif()

# Imported targets
include("${CMAKE_CURRENT_LIST_DIR}/protobuf-targets.cmake")

# protobuf-generate function
include("${CMAKE_CURRENT_LIST_DIR}/protobuf-generate.cmake")

# CMake FindProtobuf module compatible file
if(protobuf_MODULE_COMPATIBLE)
  include("${CMAKE_CURRENT_LIST_DIR}/protobuf-module.cmake")
endif()
