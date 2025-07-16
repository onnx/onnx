cmake_minimum_required(VERSION 3.5)

# The `cmake_minimum_required(VERSION 3.5...3.29)` syntax does not work with
# some versions of VS that have a patched CMake 3.11. This forces us to emulate
# the behavior using the following workaround:
if(${CMAKE_VERSION} VERSION_LESS 3.29)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.29)
endif()

project(test_installed_embed CXX)

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

add_executable(test_installed_embed ../embed.cpp)
target_link_libraries(test_installed_embed PRIVATE pybind11::embed)
set_target_properties(test_installed_embed PROPERTIES OUTPUT_NAME test_cmake_build)

# Do not treat includes from IMPORTED target as SYSTEM (Python headers in pybind11::embed).
# This may be needed to resolve header conflicts, e.g. between Python release and debug headers.
set_target_properties(test_installed_embed PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)

add_custom_target(
  check_installed_embed
  $<TARGET_FILE:test_installed_embed> ${PROJECT_SOURCE_DIR}/../test.py
  DEPENDS test_installed_embed)
