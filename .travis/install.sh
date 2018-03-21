#!/bin/bash

script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
source "${script_path%/*}/setup.sh"

time ONNX_NAMESPACE=ONNX_NAMESPACE_FOO_BAR_FOR_CI pip install -v .
