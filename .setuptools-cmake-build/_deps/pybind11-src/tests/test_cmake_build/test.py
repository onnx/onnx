from __future__ import annotations

import sys

import test_cmake_build

assert isinstance(__file__, str)  # Test this is properly set

assert test_cmake_build.add(1, 2) == 3
print(f"{sys.argv[1]} imports, runs, and adds: 1 + 2 = 3")
