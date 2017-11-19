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

# check auto-gen files up-to-date
python onnx/defs/gen_doc.py -o docs/Operators.md
python onnx/gen_proto.py
backend-test-tools generate-data
git diff --exit-code

# install onnx-ml
cd $onnx_dir
ccache -z
pip install -v . --install-option="--onnxml=1"
ccache -s

# run onnx tests again
cd $onnx_dir
pytest

# check auto-gen files up-to-date again
python onnx/defs/gen_doc.py -o docs/Operators.md
python onnx/gen_proto.py
backend-test-tools generate-data
git diff --exit-code
