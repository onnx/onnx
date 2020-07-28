#!/bin/bash

set -ex

source /tmp/venv/bin/activate

# install new libturbojpeg for pytorch/vision
wget http://ftp.br.debian.org/debian/pool/main/libj/libjpeg-turbo/libturbojpeg0_2.0.5-1_amd64.deb
sudo apt install ./libturbojpeg0_2.0.5-1_amd64.deb -y

# install torchvision from master
# the one on pypi requires cuda
pip install -q git+https://github.com/pytorch/vision.git

cd /tmp/pytorch
CI=1 exec "scripts/onnx/test.sh"
