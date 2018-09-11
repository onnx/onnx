include (ExternalProject)

set(googletest_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/include)
set(googletest_URL https://github.com/google/googletest.git)
set(googletest_BUILD ${CMAKE_CURRENT_BINARY_DIR}/googletest/)
set(googletest_TAG 0fe96607d85cf3a25ac40da369db62bbee2939a5)
#718fd88d8f145c63b8cc134cf8fed92743cc112f

if(WIN32)
  set(googletest_STATIC_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/Release/gtest.lib)
else()
  set(googletest_STATIC_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/libgtest.a)
endif()

ExternalProject_Add(googletest
    PREFIX googletest
    GIT_REPOSITORY ${googletest_URL}
    GIT_TAG ${googletest_TAG}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release --target gtest
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DBUILD_GMOCK:BOOL=OFF
        -DBUILD_GTEST:BOOL=ON
        -Dgtest_force_shared_crt:BOOL=OFF
)
