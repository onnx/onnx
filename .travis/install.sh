#!/bin/bash

script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
source "${script_path%/*}/setup.sh"

export ONNX_BUILD_TESTS=1
pip install protobuf numpy
time CMAKE_ARGS="-DONNX_WERROR=ON" ONNX_NAMESPACE=ONNX_NAMESPACE_FOO_BAR_FOR_CI python  setup.py bdist_wheel --universal --dist-dir .
find . -maxdepth 1 -name "*.whl" -ls -exec pip install {} \;
