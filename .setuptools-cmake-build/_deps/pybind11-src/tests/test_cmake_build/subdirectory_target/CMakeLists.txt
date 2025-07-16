cmake_minimum_required(VERSION 3.5)

# The `cmake_minimum_required(VERSION 3.5...3.29)` syntax does not work with
# some versions of VS that have a patched CMake 3.11. This forces us to emulate
# the behavior using the following workaround:
if(${CMAKE_VERSION} VERSION_LESS 3.29)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.29)
endif()

project(test_subdirectory_target CXX)

# Allow PYTHON_EXECUTABLE if in FINDPYTHON mode and building pybind11's tests
# (makes transition easier while we support both modes).
if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python_EXECUTABLE)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

add_subdirectory("${pybind11_SOURCE_DIR}" pybind11)

add_library(test_subdirectory_target MODULE ../main.cpp)
set_target_properties(test_subdirectory_target PROPERTIES OUTPUT_NAME test_cmake_build)

target_link_libraries(test_subdirectory_target PRIVATE pybind11::module)

# Make sure result is, for example, test_installed_target.so, not libtest_installed_target.dylib
pybind11_extension(test_subdirectory_target)

if(DEFINED Python_EXECUTABLE)
  set(_Python_EXECUTABLE "${Python_EXECUTABLE}")
elseif(DEFINED PYTHON_EXECUTABLE)
  set(_Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
else()
  message(FATAL_ERROR "No Python executable defined (should not be possible at this stage)")
endif()

add_custom_target(
  check_subdirectory_target
  ${CMAKE_COMMAND}
  -E
  env
  PYTHONPATH=$<TARGET_FILE_DIR:test_subdirectory_target>
  ${_Python_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/../test.py
  ${PROJECT_NAME}
  DEPENDS test_subdirectory_target)
