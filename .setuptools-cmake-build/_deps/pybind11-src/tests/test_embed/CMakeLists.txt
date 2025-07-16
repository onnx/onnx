possibly_uninitialized(PYTHON_MODULE_EXTENSION Python_INTERPRETER_ID)

if("${PYTHON_MODULE_EXTENSION}" MATCHES "pypy" OR "${Python_INTERPRETER_ID}" STREQUAL "PyPy")
  message(STATUS "Skipping embed test on PyPy")
  add_custom_target(cpptest) # Dummy target on PyPy. Embedding is not supported.
  set(_suppress_unused_variable_warning "${DOWNLOAD_CATCH}")
  return()
endif()

if(TARGET Python::Module AND NOT TARGET Python::Python)
  message(STATUS "Skipping embed test since no embed libs found")
  add_custom_target(cpptest) # Dummy target since embedding is not supported.
  set(_suppress_unused_variable_warning "${DOWNLOAD_CATCH}")
  return()
endif()

find_package(Catch 2.13.9)

if(CATCH_FOUND)
  message(STATUS "Building interpreter tests using Catch v${CATCH_VERSION}")
else()
  message(STATUS "Catch not detected. Interpreter tests will be skipped. Install Catch headers"
                 " manually or use `cmake -DDOWNLOAD_CATCH=ON` to fetch them automatically.")
  return()
endif()

find_package(Threads REQUIRED)

add_executable(test_embed catch.cpp test_interpreter.cpp)
pybind11_enable_warnings(test_embed)

target_link_libraries(test_embed PRIVATE pybind11::embed Catch2::Catch2 Threads::Threads)

if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
  file(COPY test_interpreter.py test_trampoline.py DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()

add_custom_target(
  cpptest
  COMMAND "$<TARGET_FILE:test_embed>"
  DEPENDS test_embed
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

pybind11_add_module(external_module THIN_LTO external_module.cpp)
set_target_properties(external_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                 "${CMAKE_CURRENT_BINARY_DIR}")
foreach(config ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${config} config)
  set_target_properties(external_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${config}
                                                   "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()
add_dependencies(cpptest external_module)

add_dependencies(check cpptest)
