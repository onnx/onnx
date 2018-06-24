include (ExternalProject)

set(googletest_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/include)
set(googletest_URL https://github.com/google/googletest.git)
set(googletest_BUILD ${CMAKE_CURRENT_BINARY_DIR}/googletest/)
set(googletest_TAG 0fe96607d85cf3a25ac40da369db62bbee2939a5)
#718fd88d8f145c63b8cc134cf8fed92743cc112f

if(MSVC)
    if("${CMAKE_GENERATOR}" MATCHES "Ninja")
        set(googletest_STATIC_LIBRARIES
            ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/gtest.lib)
    else()
        set(googletest_STATIC_LIBRARIES
            ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/Release/gtest.lib)
    endif()
    set(ADDITIONAL_C_FLAGS "/wd4996")
    set(ADDITIONAL_CXX_FLAGS "/wd4996")
    IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        if (PY_ARCH)
            set(ADDITIONAL_C_FLAGS "${ADDITIONAL_C_FLAGS} -m${PY_ARCH}")
            set(ADDITIONAL_CXX_FLAGS "${ADDITIONAL_CXX_FLAGS} -m${PY_ARCH}")
        endif(PY_ARCH)
    endif()
else()
  set(googletest_STATIC_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/libgtest.a)
  set(ADDITIONAL_C_FLAGS "")
  set(ADDITIONAL_CXX_FLAGS "")
endif()

MESSAGE(STATUS "ADDITIONAL_C_FLAGS: ${ADDITIONAL_C_FLAGS}")
MESSAGE(STATUS "ADDITIONAL_CXX_FLAGS: ${ADDITIONAL_CXX_FLAGS}")

ExternalProject_Add(googletest
    PREFIX googletest
    GIT_REPOSITORY ${googletest_URL}
    GIT_TAG ${googletest_TAG}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release --target gtest
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS "${googletest_STATIC_LIBRARIES}"
    CMAKE_ARGS
        -DCMAKE_C_FLAGS=${ADDITIONAL_C_FLAGS}
        -DCMAKE_CXX_FLAGS=${ADDITIONAL_CXX_FLAGS}
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DBUILD_GMOCK:BOOL=OFF
        -DBUILD_GTEST:BOOL=ON
        -Dgtest_force_shared_crt:BOOL=OFF
)
