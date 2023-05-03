#!/bin/bash

# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

export CORE_NUMBER=$1
export INSTALL_PROTOBUF_PATH=$2
export BUILD_TYPE=$3

if [[ -z "$CORE_NUMBER" ]]; then
   export CORE_NUMBER=1
fi

if [[ -z "$INSTALL_PROTOBUF_PATH" ]]; then
   export INSTALL_PROTOBUF_PATH=/usr
fi

if [[ -z "$BUILD_TYPE" ]]; then
   export BUILD_TYPE=Release
fi

# Build protobuf from source with -fPIC on Unix-like system
ORIGINAL_PATH=$(pwd)
cd ..
set -e
wget -c https://github.com/protocolbuffers/protobuf/releases/download/v22.3/protobuf-22.3.tar.gz
tar -xvf protobuf-22.3.tar.gz
pushd protobuf-22.3/third_party/abseil-cpp/
# Reference: https://github.com/abseil/abseil-cpp/releases/tag/20230125.0
wget https://github.com/abseil/abseil-cpp/archive/refs/tags/20230125.0.tar.gz -O 20230125.0.tar.gz
tar -xvf 20230125.0.tar.gz
mv abseil-cpp-20230125.0/* . && rm -rf abseil-cpp-20230125.0/
popd
cd protobuf-22.3
set +e
mkdir build_source && cd build_source
cmake ../cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_PROTOBUF_PATH -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_CXX_STANDARD=14 -DCMAKE_BUILD_TYPE=$BUILD_TYPE
make -j$CORE_NUMBER
if [ "$INSTALL_PROTOBUF_PATH" == "/usr" ]; then
    # install protobuf on default system path so it needs sudo permission
    sudo make install
else
    make install
    echo "INSTALL_PROTOBUF_PATH=$INSTALL_PROTOBUF_PATH"
    export PATH=$INSTALL_PROTOBUF_PATH/include:$INSTALL_PROTOBUF_PATH/lib:$INSTALL_PROTOBUF_PATH/bin:$PATH
fi
cd $ORIGINAL_PATH
