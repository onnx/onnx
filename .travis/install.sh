#!/bin/bash

script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
source "${script_path%/*}/setup.sh"

export ONNX_BUILD_TESTS=1
pip install --quiet protobuf numpy

export CMAKE_ARGS="-DONNX_WERROR=ON"
if [[ -n "USE_LITE_PROTO" ]]; then
    export CMAKE_ARGS="${CMAKE_ARGS} -DONNX_USE_LITE_PROTO=ON"
fi
export CMAKE_ARGS="${CMAKE_ARGS} -DONNXIFI_DUMMY_BACKEND=ON"
export ONNX_NAMESPACE=ONNX_NAMESPACE_FOO_BAR_FOR_CI

time python setup.py --quiet bdist_wheel --universal --dist-dir .
find . -maxdepth 1 -name "*.whl" -ls -exec pip install {} \;
