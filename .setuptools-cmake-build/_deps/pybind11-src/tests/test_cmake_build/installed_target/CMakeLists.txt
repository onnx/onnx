cmake_minimum_required(VERSION 3.5)

# The `cmake_minimum_required(VERSION 3.5...3.29)` syntax does not work with
# some versions of VS that have a patched CMake 3.11. This forces us to emulate
# the behavior using the following workaround:
if(${CMAKE_VERSION} VERSION_LESS 3.29)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.29)
endif()

project(test_installed_target CXX)

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

add_library(test_installed_target MODULE ../main.cpp)

target_link_libraries(test_installed_target PRIVATE pybind11::module)
set_target_properties(test_installed_target PROPERTIES OUTPUT_NAME test_cmake_build)

# Make sure result is, for example, test_installed_target.so, not libtest_installed_target.dylib
pybind11_extension(test_installed_target)

# Do not treat includes from IMPORTED target as SYSTEM (Python headers in pybind11::module).
# This may be needed to resolve header conflicts, e.g. between Python release and debug headers.
set_target_properties(test_installed_target PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)

if(DEFINED Python_EXECUTABLE)
  set(_Python_EXECUTABLE "${Python_EXECUTABLE}")
elseif(DEFINED PYTHON_EXECUTABLE)
  set(_Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
else()
  message(FATAL_ERROR "No Python executable defined (should not be possible at this stage)")
endif()

add_custom_target(
  check_installed_target
  ${CMAKE_COMMAND}
  -E
  env
  PYTHONPATH=$<TARGET_FILE_DIR:test_installed_target>
  ${_Python_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/../test.py
  ${PROJECT_NAME}
  DEPENDS test_installed_target)
