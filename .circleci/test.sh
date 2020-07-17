#!/bin/bash

set -ex

source /tmp/venv/bin/activate

wget https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py
python collect_env.py

# install torchvision from master
# the one on pypi requires cuda
pip install numpy
pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html
pip install -q git+https://github.com/pytorch/vision.git

cd /tmp/pytorch
CI=1 exec "scripts/onnx/test.sh"
