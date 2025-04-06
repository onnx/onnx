# Target system
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR ARM64)

# Compiler paths
set(CMAKE_C_COMPILER C:/mingw64/bin/arm64-unknown-linux-gnu-gcc.exe)
set(CMAKE_CXX_COMPILER C:/mingw64/bin/arm64-unknown-linux-gnu-g++.exe)

# Path to the cross-compilation toolchain
set(CMAKE_FIND_ROOT_PATH C:/cross-toolchain)

# Search for programs in the host system
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers in the target system
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Additional flags
set(CMAKE_C_FLAGS "-O2")
set(CMAKE_CXX_FLAGS "-O2")