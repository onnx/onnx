#!/bin/bash

scripts_dir=$(dirname $(readlink -e "${BASH_SOURCE[0]}"))
source "$scripts_dir/common";

onnx_dir="$PWD"

# install onnx
cd $onnx_dir
ccache -z
pip install .
ccache -s

# onnx tests
cd $onnx_dir
pip install pytest-cov
pytest

# check auto-gen files up-to-date
# docs/Operators.md
python onnx/defs/gen_doc.py -o docs/Operators.md
git diff --exit-code docs/Operators.md
