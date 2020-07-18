#!/bin/bash

set -ex

source /tmp/venv/bin/activate

# update libpng for pytorch/vision
sudo apt-get update -y
sudo apt-get install -y libpng16-16

# install torchvision from master
# the one on pypi requires cuda
# Currently the master branch of pytorch/vision is not stable
# Should set it back to the master if the issue from master branch has been resovled
# Therefore use certain commit on 07/06/2020 for
git clone https://github.com/pytorch/vision.git
cd vision
git submodule update --init --recursive
python setup.py develop
cd ..

cd /tmp/pytorch
CI=1 exec "scripts/onnx/test.sh"
