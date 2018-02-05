#!/bin/bash

scripts_dir=$(dirname $(readlink -e "${BASH_SOURCE[0]}"))
source "$scripts_dir/common";

onnx_dir="$PWD"

# install onnx
cd $onnx_dir
ccache -z
pip install -v .
ccache -s

# onnx tests
cd $onnx_dir
pip install pytest-cov nbval
pytest

# check line endings to be UNIX
find . -type f -regextype posix-extended -regex '.*\.(py|cpp|md|h|cc|proto|proto3|in)' | xargs dos2unix
git status
git diff --exit-code

# check auto-gen files up-to-date
python onnx/defs/gen_doc.py
python onnx/gen_proto.py
backend-test-tools generate-data
git status
git diff --exit-code

# Do not hardcode onnx's namespace in the c++ source code, so that
# other libraries who statically link with onnx can hide onnx symbols
# in a private namespace.
! grep -R --include='*.cc' --include='*.h' 'namespace onnx' .
! grep -R --include='*.cc' --include='*.h' 'onnx::' .
