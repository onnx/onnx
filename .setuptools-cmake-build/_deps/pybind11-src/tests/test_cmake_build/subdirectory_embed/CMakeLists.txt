cmake_minimum_required(VERSION 3.5)

# The `cmake_minimum_required(VERSION 3.5...3.29)` syntax does not work with
# some versions of VS that have a patched CMake 3.11. This forces us to emulate
# the behavior using the following workaround:
if(${CMAKE_VERSION} VERSION_LESS 3.29)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.29)
endif()

project(test_subdirectory_embed CXX)

set(PYBIND11_INSTALL
    ON
    CACHE BOOL "")
set(PYBIND11_EXPORT_NAME test_export)

# Allow PYTHON_EXECUTABLE if in FINDPYTHON mode and building pybind11's tests
# (makes transition easier while we support both modes).
if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python_EXECUTABLE)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

add_subdirectory("${pybind11_SOURCE_DIR}" pybind11)

# Test basic target functionality
add_executable(test_subdirectory_embed ../embed.cpp)
target_link_libraries(test_subdirectory_embed PRIVATE pybind11::embed)
set_target_properties(test_subdirectory_embed PROPERTIES OUTPUT_NAME test_cmake_build)

add_custom_target(
  check_subdirectory_embed
  $<TARGET_FILE:test_subdirectory_embed> "${PROJECT_SOURCE_DIR}/../test.py"
  DEPENDS test_subdirectory_embed)

# Test custom export group -- PYBIND11_EXPORT_NAME
add_library(test_embed_lib ../embed.cpp)
target_link_libraries(test_embed_lib PRIVATE pybind11::embed)

install(
  TARGETS test_embed_lib
  EXPORT test_export
  ARCHIVE DESTINATION bin
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib)
install(EXPORT test_export DESTINATION lib/cmake/test_export/test_export-Targets.cmake)
