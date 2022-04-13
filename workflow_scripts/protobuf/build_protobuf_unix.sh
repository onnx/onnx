#!/bin/bash

# SPDX-License-Identifier: Apache-2.0

export CORE_NUMBER=$1
export INSTALL_PROTOBUF_PATH=$2

if [[ -z "$CORE_NUMBER" ]]; then
   export CORE_NUMBER=1
fi

if [[ -z "$INSTALL_PROTOBUF_PATH" ]]; then
   export INSTALL_PROTOBUF_PATH=/usr
fi

# Build protobuf from source with -fPIC on Unix-like system
ORIGINAL_PATH=$(pwd)
cd ..
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.16.0/protobuf-cpp-3.16.0.tar.gz
tar -xvf protobuf-cpp-3.16.0.tar.gz
cd protobuf-3.16.0
mkdir build_source && cd build_source
cmake ../cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_PROTOBUF_PATH -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -j$CORE_NUMBER
if [ "$INSTALL_PROTOBUF_PATH" == "/usr" ]; then
    # install protobuf on default system path so it needs sudo permission
    sudo make install
else
    make install
    export PATH=$INSTALL_PROTOBUF_PATH/include:$INSTALL_PROTOBUF_PATH/lib:$INSTALL_PROTOBUF_PATH/bin:$PATH
fi
cd $ORIGINAL_PATH
