#!/bin/bash

script_path=$(python -c "import os; import sys; print(os.path.realpath(sys.argv[1]))" "${BASH_SOURCE[0]}")
source "${script_path%/*}/setup.sh"

# onnx c++ API tests
export LD_LIBRARY_PATH="${top_dir}/.setuptools-cmake-build/:$LD_LIBRARY_PATH"
# do not use find -exec here, it would ignore the segement fault of gtest.
./.setuptools-cmake-build/onnx_gtests
./.setuptools-cmake-build/onnxifi_test_driver_gtests onnx/backend/test/data/node

# onnx python API tests
pip install --quiet pytest nbval
pytest

# lint python code
pip install --quiet flake8
flake8

# Mypy only works with Python 3
if [ "${PYTHON_VERSION}" != "python2" ]; then
  # Mypy only works with our generated _pb.py files when we install in develop mode, so let's do that
  pip uninstall -y onnx
  time ONNX_NAMESPACE=ONNX_NAMESPACE_FOO_BAR_FOR_CI pip install -e .[mypy]

  time python setup.py --quiet typecheck

  pip uninstall -y onnx
  rm -rf .setuptools-cmake-build
  time ONNX_NAMESPACE=ONNX_NAMESPACE_FOO_BAR_FOR_CI pip install .
fi

# check line endings to be UNIX
find . -type f -regextype posix-extended -regex '.*\.(py|cpp|md|h|cc|proto|proto3|in)' | xargs dos2unix --quiet
git status
git diff --exit-code

# check auto-gen files up-to-date
python onnx/defs/gen_doc.py
python onnx/gen_proto.py
python onnx/backend/test/stat_coverage.py
backend-test-tools generate-data
git status
git diff --exit-code

# Do not hardcode onnx's namespace in the c++ source code, so that
# other libraries who statically link with onnx can hide onnx symbols
# in a private namespace.
! grep -R --include='*.cc' --include='*.h' 'namespace onnx' .
! grep -R --include='*.cc' --include='*.h' 'onnx::' .
