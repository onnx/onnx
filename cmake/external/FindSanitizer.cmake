# SPDX-License-Identifier: Apache-2.0

# Find sanitizers
#
# This module sets the following targets:
#  Sanitizer::address
#  Sanitizer::thread
#  Sanitizer::undefined
#  Sanitizer::memory
include_guard(GLOBAL)

option(UBSAN_FLAGS "additional UBSAN flags" OFF)
option(MSAN_FLAGS "additional MSAN flags" OFF)

get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)

set(_source_code
    [==[
  #include <stdio.h>
  int main() {
  printf("hello world!");
  return 0;
  }
  ]==])

set(_bug_address_code
    [==[
#include <stdlib.h>
int main(int argc, char **argv) {
  int *array = (int*)malloc(100*sizeof(int));
  array[0] = 0;
  int res = array[argc + 100];  // BOOM
  free(array);
  return res;
}
]==])

set(_bug_undefined_code
    [==[
int main(int argc, char **argv) {
  int k = 0x7fffffff;
  k += argc;
  return 0;
}
]==])

include(CMakePushCheckState)
foreach(lang IN LISTS languages)
  if(lang STREQUAL C)
    include(CheckCSourceCompiles)
    include(CheckCSourceRuns)
  elseif(lang STREQUAL CXX)
    include(CheckCXXSourceCompiles)
    include(CheckCXXSourceRuns)
  else()
    continue()
  endif()
  foreach(sanitizer_name IN ITEMS address thread undefined memory)
    if(TARGET Sanitizer::${sanitizer_name}_${lang})
      continue()
    endif()
    if(CMAKE_${lang}_COMPILER_ID STREQUAL "MSVC")
      if(sanitizer_name STREQUAL "address")
        set(SANITIZER_FLAGS "/fsanitize=${sanitizer_name}")
      else()
        continue()
      endif()
    else()
      set(SANITIZER_FLAGS
          "-fsanitize=${sanitizer_name};-fno-omit-frame-pointer")
    endif()
    if(sanitizer_name STREQUAL "undefined" AND UBSAN_FLAGS)
      list(APPEND SANITIZER_FLAGS "${UBSAN_FLAGS}")
    endif()
    if(sanitizer_name STREQUAL "memory")
      list(APPEND SANITIZER_FLAGS "-fsanitize-memory-track-origins=2")
      if(MSAN_FLAGS)
        list(APPEND SANITIZER_FLAGS "${MSAN_FLAGS}")
      endif()
    endif()
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_QUIET ON)
    string(REPLACE ";" " " CMAKE_REQUIRED_FLAGS "${SANITIZER_FLAGS}")

    set(SANITIZER_LINK_FLAGS)
    if(CMAKE_${lang}_COMPILER_ID STREQUAL "MSVC")
      list(APPEND SANITIZER_LINK_FLAGS "/INCREMENTAL:NO")
    else()
      list(APPEND SANITIZER_LINK_FLAGS "-fsanitize=${sanitizer_name}")
    endif()
    set(CMAKE_REQUIRED_LINK_OPTIONS "${SANITIZER_LINK_FLAGS}")

    unset(__res CACHE)
    if(lang STREQUAL C)
      if(CMAKE_${lang}_COMPILER_ID STREQUAL "MSVC")
        check_c_source_compiles("${_source_code}" __res)
      else()
        check_c_source_runs("${_source_code}" __res)
      endif()
    else()
      if(CMAKE_${lang}_COMPILER_ID STREQUAL "MSVC")
        check_cxx_source_compiles("${_source_code}" __res)
      else()
        check_cxx_source_runs("${_source_code}" __res)
      endif()
    endif()
    if(NOT __res)
      message(WARNING "Can't find ${sanitizer_name} in ${lang}")
      cmake_pop_check_state()
      continue()
    endif()

    unset(__res CACHE)
    if(NOT CMAKE_${lang}_COMPILER_ID STREQUAL "MSVC" AND (sanitizer_name STREQUAL "address") OR (sanitizer_name STREQUAL "undefined"))
      set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -fno-sanitize-recover=all")
      if(lang STREQUAL C)
        check_c_source_runs("${_bug_${sanitizer_name}_code}" __res)
      else()
        check_cxx_source_runs("${_bug_${sanitizer_name}_code}" __res)
      endif()
      if(__res)
        message(WARNING "Buffer overflow bug is not detected in ${lang} ${sanitizer_name}")
        cmake_pop_check_state()
        continue()
      endif()
    endif()

    add_library(Sanitizer::${sanitizer_name}_${lang} INTERFACE IMPORTED GLOBAL)
    if(NOT TARGET Sanitizer::${sanitizer_name})
      add_library(Sanitizer::${sanitizer_name} INTERFACE IMPORTED GLOBAL)
    endif()
    target_link_libraries(Sanitizer::${sanitizer_name}
                          INTERFACE Sanitizer::${sanitizer_name}_${lang})
    foreach(SANITIZER_FLAG IN LISTS SANITIZER_FLAGS)
      target_compile_options(
        Sanitizer::${sanitizer_name}_${lang}
        INTERFACE $<$<COMPILE_LANGUAGE:${lang}>:${SANITIZER_FLAG}>)
    endforeach()
    foreach(SANITIZER_FLAG IN LISTS SANITIZER_LINK_FLAGS)
      target_link_options(Sanitizer::${sanitizer_name}_${lang} INTERFACE
                          $<$<COMPILE_LANGUAGE:${lang}>:${SANITIZER_FLAG}>)
    endforeach()

    if(CMAKE_${lang}_COMPILER_ID STREQUAL "Clang")
      target_compile_options(
        Sanitizer::${sanitizer_name}_${lang}
        INTERFACE $<$<COMPILE_LANGUAGE:${lang}>:-shared-libsan>)
    endif()

    if(sanitizer_name STREQUAL "address" AND lang STREQUAL CXX)
      if(CMAKE_${lang}_COMPILER_ID STREQUAL "MSVC")
        target_compile_definitions(
          Sanitizer::${sanitizer_name}_${lang}
          INTERFACE $<$<COMPILE_LANGUAGE:${lang}>:_DISABLE_VECTOR_ANNOTATION>
                    $<$<COMPILE_LANGUAGE:${lang}>:_DISABLE_STRING_ANNOTATION>)
      else()
        target_compile_definitions(
          Sanitizer::${sanitizer_name}_${lang}
          INTERFACE
            $<$<COMPILE_LANGUAGE:${lang}>:_GLIBCXX_SANITIZE_VECTOR>
            $<$<COMPILE_LANGUAGE:${lang}>:_GLIBCXX_SANITIZE_STD_ALLOCATOR>)
      endif()
    endif()
    cmake_pop_check_state()
  endforeach()
endforeach()
