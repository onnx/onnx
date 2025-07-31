#!/bin/bash

# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

set -e -x

# CLI arguments
PY_VERSION=$1
PLAT=$2
BUILD_MODE=$3  # build mode (release or preview)
SOURCE_DATE_EPOCH_ARG=$4 # https://reproducible-builds.org/docs/source-date-epoch/

# Set SOURCE_DATE_EPOCH for reproducible builds
if [ -n "$SOURCE_DATE_EPOCH_ARG" ]; then
    export SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH_ARG
fi

echo "SOURCE_DATE_EPOCH: $SOURCE_DATE_EPOCH"
echo "Python version: $PY_VERSION"
echo "Platform: $PLAT"
echo "Build mode: $BUILD_MODE"

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

declare -A python_map=(["3.9"]="cp39-cp39" ["3.10"]="cp310-cp310" ["3.11"]="cp311-cp311" ["3.12"]="cp312-cp312" ["3.13"]="cp313-cp313" ["3.13t"]="cp313-cp313t" ["3.14-dev"]="cp314-cp314")
PY_VER=${python_map[$PY_VERSION]}
PIP_INSTALL_COMMAND="/opt/python/${PY_VER}/bin/pip install --only-binary google-re2 --no-cache-dir -q"
PYTHON_COMMAND="/opt/python/${PY_VER}/bin/python"

# Update pip
$PIP_INSTALL_COMMAND --upgrade pip
$PIP_INSTALL_COMMAND cmake

# Build protobuf from source
yum install -y wget
source workflow_scripts/protobuf/build_protobuf_unix.sh "$(nproc)" "$(pwd)"/protobuf/protobuf_install

# set ONNX build environments
export ONNX_ML=1
export CMAKE_ARGS="-DONNX_USE_LITE_PROTO=ON"

$PIP_INSTALL_COMMAND -v -r requirements-release_build.txt || { echo "Installing Python requirements failed."; exit 1; }

if [ "$BUILD_MODE" != "release" ]; then
    echo "Building preview wheels..."
    sed -i 's/name = "onnx"/name = "onnx-weekly"/' 'pyproject.toml'
    export ONNX_PREVIEW_BUILD=1
else
    echo "Building release wheels..."
fi

# Build the wheels
if ! $PYTHON_COMMAND -m build --wheel; then
    echo "Building wheels failed."
    exit 1
fi

# Bundle external shared libraries into the wheels
# find -exec does not preserve failed exit codes, so use an output file for failures
failed_wheels=$PWD/failed-wheels
rm -f "$failed_wheels"
find . -type f -iname "*-linux*.whl" -exec sh -c "auditwheel repair '{}' -w \$(dirname '{}') --plat '${PLAT}' || { echo 'Repairing wheels failed.'; auditwheel show '{}' >> '$failed_wheels'; }" \;

if [[ -f "$failed_wheels" ]]; then
    echo "Repairing wheels failed:"
    cat failed-wheels
    exit 1
fi

# Remove useless *-linux*.whl; only keep manylinux*.whl
rm -f dist/*-linux*.whl

echo "Successfully build wheels:"
find . -type f -iname "*manylinux*.whl"
