#!/bin/bash

scripts_dir=$(dirname $(readlink -e "${BASH_SOURCE[0]}"))
source "$scripts_dir/common";

onnx_dir="$PWD"

# install onnx
install_option=()

if [[ $ONNX_ML == true ]]; then
    install_option+=(--install-option="--onnxml=1")
fi

cd $onnx_dir
ccache -z
pip install -v . "${install_option[@]}"
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
