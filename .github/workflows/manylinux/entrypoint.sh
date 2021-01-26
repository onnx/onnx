#!/bin/bash

set -e -x

# CLI arguments
PY_VERSIONS=$1
PLAT=$2
BUILD_REQUIREMENTS='numpy==1.16.6 protobuf==3.11.3'
SYSTEM_PACKAGES='cmake3'

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

if [ ! -z "$SYSTEM_PACKAGES" ]; then
    yum install -y ${SYSTEM_PACKAGES}  || { echo "Installing yum package(s) failed."; exit 1; }
fi

# Build protobuf
export NUM_PROCESSOR=`grep -c ^processor /proc/cpuinfo`

ONNX_PATH=$(pwd)
cd ..
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout 3.11.x
git submodule update --init --recursive
./autogen.sh --disable-shared --enable-pic

CFLAGS="-fPIC -g -O2" CXXFLAGS="-fPIC -g -O2" ./configure --disable-shared
make -j${NUM_PROCESSOR}
make check
make install
ldconfig
cd $ONNX_PATH

# Compile wheels
# Need to be updated if there is a new Python Version
declare -A python_map=( ["3.5"]="cp35-cp35m" ["3.6"]="cp36-cp36m" ["3.7"]="cp37-cp37m" ["3.8"]="cp38-cp38" ["3.9"]="cp39-cp39")
declare -A python_include=( ["3.5"]="3.5m" ["3.6"]="3.6m" ["3.7"]="3.7m" ["3.8"]="3.8" ["3.9"]="3.9")
PY_VER=${python_map[$PY_VERSIONS]}

# set ONNX build environments
export ONNX_BUILD_TESTS=1
export USE_MSVC_STATIC_RUNTIME=1
export ONNX_ML=1
export CMAKE_ARGS="-DONNX_USE_LITE_PROTO=ON -DPYTHON_INCLUDE_DIR=/opt/python/${PY_VER}/include/python${python_include[$PY_VERSIONS]} -DPYTHON_LIBRARY=/usr/lib64/librt.so"

# Update pip
/opt/python/"${PY_VER}"/bin/pip install --upgrade --no-cache-dir pip

# Check if requirements were passed
if [ ! -z "$BUILD_REQUIREMENTS" ]; then
    /opt/python/"${PY_VER}"/bin/pip install --no-cache-dir ${BUILD_REQUIREMENTS} || { echo "Installing requirements failed."; exit 1; }
fi

# Build wheels
if [ "$PLAT" = "manylinux2010_i686" ]; then
    /opt/python/"${PY_VER}"/bin/pip wheel . -w ./dist_i686 --no-deps || { echo "Building wheels failed."; exit 1; }
else
    /opt/python/"${PY_VER}"/bin/pip wheel . -w ./dist --no-deps || { echo "Building wheels failed."; exit 1; }
fi

echo "Succesfully build wheels:"
find . -type f -iname "*-manylinux*.whl"
