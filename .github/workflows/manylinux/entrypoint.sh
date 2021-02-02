#!/bin/bash

set -e -x

# CLI arguments
PY_VERSION=$1
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
PY_VER=${python_map[$PY_VERSION]}

# set ONNX build environments
export ONNX_ML=1
export CMAKE_ARGS="-DPYTHON_INCLUDE_DIR=/opt/python/${PY_VER}/include/python${python_include[$PY_VERSION]}"

# Update pip
/opt/python/"${PY_VER}"/bin/pip install --upgrade --no-cache-dir pip

# Check if requirements were passed
if [ ! -z "$BUILD_REQUIREMENTS" ]; then
    /opt/python/"${PY_VER}"/bin/pip install --no-cache-dir ${BUILD_REQUIREMENTS} || { echo "Installing requirements failed."; exit 1; }
fi

# Build wheels
/opt/python/"${PY_VER}"/bin/pip wheel . -w ./dist --no-deps || { echo "Building wheels failed."; exit 1; }

# Bundle external shared libraries into the wheels
# find -exec does not preserve failed exit codes, so use an output file for failures
failed_wheels=$PWD/failed-wheels
rm -f "$failed_wheels"
find . -type f -iname "*-linux*.whl" -exec sh -c "auditwheel repair '{}' -w \$(dirname '{}') --plat '${PLAT}' || { echo 'Repairing wheels failed.'; auditwheel show '{}' >> "$failed_wheels"; }" \;

if [[ -f "$failed_wheels" ]]; then
    echo "Repairing wheels failed:"
    cat failed-wheels
    exit 1
fi

# Remove useless *-linux*.whl; only keep -manylinux*.whl
rm -f dist/*-linux*.whl

echo "Succesfully build wheels:"
find . -type f -iname "*-manylinux*.whl"