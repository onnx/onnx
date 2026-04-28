#!/bin/bash
# shellcheck disable=SC2164,SC2103,SC2086

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
wget https://github.com/abseil/abseil-cpp/releases/download/20230802.2/abseil-cpp-20230802.2.tar.gz
tar -xvf abseil-cpp-20230802.2.tar.gz

wget https://github.com/protocolbuffers/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz
tar -xvf protobuf-25.1.tar.gz
cd protobuf-25.1
mkdir build_source && cd build_source
cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL_PROTOBUF_PATH -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DABSL_ROOT_DIR="${ORIGINAL_PATH}/../abseil-cpp-20230802.2" -DCMAKE_CXX_STANDARD=17 -DABSL_PROPAGATE_CXX_STD=on ..
if [ "$INSTALL_PROTOBUF_PATH" == "/usr" ]; then
    # Don't use sudo for root
    if [[ "$(id -u)" == "0" ]]; then
      cmake --build . --target install --parallel $CORE_NUMBER
    else
      # install Protobuf on default system path so it needs sudo permission
      sudo cmake --build . --target install --parallel $CORE_NUMBER
    fi
else
    cmake --build . --target install --parallel $CORE_NUMBER
    export PATH=$INSTALL_PROTOBUF_PATH/include:$INSTALL_PROTOBUF_PATH/lib:$INSTALL_PROTOBUF_PATH/bin:$PATH
fi
protoc --version
cd $ORIGINAL_PATH
