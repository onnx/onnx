#!/bin/bash

set -ex

source /tmp/venv/bin/activate

# install torchvision from master
# the one on pypi requires cuda
# Currently the master branch of pytorch/vision is not stable
# Should set it back to the master if the issue from master branch has been resovled

pip install -q git+https://github.com/pytorch/vision.git

cd /tmp/pytorch
CI=1 exec "scripts/onnx/test.sh"
