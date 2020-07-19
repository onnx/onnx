#!/bin/bash

set -ex

source /tmp/venv/bin/activate

# update libpng for pytorch/vision
wget http://archive.ubuntu.com/ubuntu/pool/universe/libp/libpng1.6/libpng16-16_1.6.20-2_amd64.deb
sudo apt install ./libpng16-16_1.6.20-2_amd64.deb

# install torchvision from master
# the one on pypi requires cuda
# Currently the master branch of pytorch/vision is not stable
# Should set it back to the master if the issue from master branch has been resovled

pip install -q git+https://github.com/pytorch/vision.git

cd /tmp/pytorch
CI=1 exec "scripts/onnx/test.sh"
