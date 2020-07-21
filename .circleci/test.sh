#!/bin/bash

set -ex

source /tmp/venv/bin/activate

# update libpng for pytorch/vision
wget http://ftp.br.debian.org/debian/pool/main/libj/libjpeg-turbo/libturbojpeg0_2.0.5-1_amd64.deb
sudo apt install ./libturbojpeg0_2.0.5-1_amd64.deb
#sudo apt install ./libturbojpeg0-dev_1.5.1-2_amd64.deb
apt list --installed | grep -E "lib(png|jpeg)"

# install torchvision from master
# the one on pypi requires cuda
# Currently the master branch of pytorch/vision is not stable
# Should set it back to the master if the issue from master branch has been resovled

pip install -vvv git+https://github.com/pytorch/vision.git

cd /tmp/pytorch
CI=1 exec "scripts/onnx/test.sh"
