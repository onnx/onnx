#!/bin/bash

set -ex

source /tmp/venv/bin/activate

# install torchvision from master
# the one on pypi requires cuda
pip install -q git+https://github.com/pytorch/vision.git@86b6c3e22e9d7d8b0fa25d08704e6a31a364973b

cd /tmp/pytorch
CI=1 exec "scripts/onnx/test.sh"
