#!/bin/bash

set -ex

source /tmp/venv/bin/activate

# install torchvision from master
# the one on pypi requires cuda
pip install -q git+https://github.com/pytorch/vision.git

cd /tmp/pytorch
CI=1 exec "scripts/onnx/test.sh"
