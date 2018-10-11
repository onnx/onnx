#!/bin/bash

set -ex

# setup onnx as the submodule of pytorch
export ONNX_DIR="$PYTORCH_DIR/third_party/onnx"
git clone --recursive --quiet https://github.com/pytorch/pytorch.git "$PYTORCH_DIR"
rm -rf "$ONNX_DIR"
cp -r "$PWD" "$ONNX_DIR"

# install ninja to speedup the build
pip install ninja

# install everything
exec "$PYTORCH_DIR/scripts/onnx/install-develop.sh"

# report sccache hit/miss stats
if hash sccache 2>/dev/null; then
    sccache --show-stats
fi
