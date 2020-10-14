#!/bin/bash
# use PYTHON_EXECUTABLE from CMake to get the right python to execute
PYTHON_EXECUTABLE=$(grep PYTHON_EXECUTABLE:FILEPATH CMakeCache.txt | cut -d "=" -f2)
$PYTHON_EXECUTABLE -u "$PWD"/../tools/protoc-gen-mypy.py