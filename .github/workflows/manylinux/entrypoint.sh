#!/bin/bash

set -e -x

# CLI arguments
PY_VERSION=$1
PLAT=$2
GITHUB_EVENT_NAME=$3
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
mkdir build_source && cd build_source

cmake ../cmake -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
make -j${NUM_PROCESSOR}
make install
ldconfig
cd $ONNX_PATH

# Compile wheels
# Need to be updated if there is a new Python Version
declare -A python_map=( ["3.6"]="cp36-cp36m" ["3.7"]="cp37-cp37m" ["3.8"]="cp38-cp38" ["3.9"]="cp39-cp39")
declare -A python_include=( ["3.6"]="3.6m" ["3.7"]="3.7m" ["3.8"]="3.8" ["3.9"]="3.9")
PY_VER=${python_map[$PY_VERSION]}
PIP_COMMAND="/opt/python/${PY_VER}/bin/pip install --no-cache-dir"
PYTHON_COMAND="/opt/python/"${PY_VER}"/bin/python"

# set ONNX build environments
export ONNX_ML=1
export CMAKE_ARGS="-DPYTHON_INCLUDE_DIR=/opt/python/${PY_VER}/include/python${python_include[$PY_VERSION]}"

# Update pip
$PIP_COMMAND --upgrade --no-cache-dir pip

# Check if requirements were passed
if [ ! -z "$BUILD_REQUIREMENTS" ]; then
    $PIP_COMMAND --no-cache-dir ${BUILD_REQUIREMENTS} || { echo "Installing requirements failed."; exit 1; }
fi

# Build wheels
if [ "$GITHUB_EVENT_NAME" == "schedule" ]; then
    $PYTHON_COMAND setup.py bdist_wheel --weekly_build || { echo "Building wheels failed."; exit 1; }
else
    $PYTHON_COMAND setup.py bdist_wheel || { echo "Building wheels failed."; exit 1; }
fi


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