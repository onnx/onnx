#!/bin/bash

set -ex

# setup onnx as the submodule of pytorch
export ONNX_DIR="$PYTORCH_DIR/third_party/onnx"
git clone --recursive --quiet https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
rm -rf "$ONNX_DIR"
cp -r "$PWD" "$ONNX_DIR"

pip install ninja

# install pytorch
cd $PYTORCH_DIR
pip install -r requirements.txt
python setup.py build develop
# install onnx
cd $ONNX_DIR
python setup.py develop

if hash sccache 2>/dev/null; then
    sccache --show-stats
endif
