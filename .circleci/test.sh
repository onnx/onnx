#!/bin/bash

set -ex

source /tmp/venv/bin/activate

# install torchvision from master
# the one on pypi requires cuda
# Currently the master branch of pytorch/vision is not stable
# Should set it back to the master if the issue from master branch has been resovled
# Therefore use certain commit on 07/06/2020 for
pip install -vvv git+https://github.com/pytorch/vision.git@86b6c3e22e9d7d8b0fa25d08704e6a31a364973

cd /tmp/pytorch
CI=1 exec "scripts/onnx/test.sh"
