#!/bin/bash

# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

set -e -x

# CLI arguments
PY_VERSION=$1
PLAT=$2
GITHUB_EVENT_NAME=$3

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib

# Compile wheels
# Need to be updated if there is a new Python Version
if [ "$(uname -m)" == "aarch64" ]; then
 PIP_INSTALL_COMMAND="$PY_VERSION -m pip install --no-cache-dir -q"
 PYTHON_COMMAND="$PY_VERSION"
else
 declare -A python_map=( ["3.8"]="cp38-cp38" ["3.9"]="cp39-cp39" ["3.10"]="cp310-cp310" ["3.11"]="cp311-cp311" ["3.12"]="cp312-cp312")
 PY_VER=${python_map[$PY_VERSION]}
 PIP_INSTALL_COMMAND="/opt/python/${PY_VER}/bin/pip install --no-cache-dir -q"
 PYTHON_COMMAND="/opt/python/${PY_VER}/bin/python"
fi

# Update pip
$PIP_INSTALL_COMMAND --upgrade pip
$PIP_INSTALL_COMMAND cmake

# Build protobuf from source
yum install -y wget
source workflow_scripts/protobuf/build_protobuf_unix.sh "$(nproc)" "$(pwd)"/protobuf/protobuf_install

# set ONNX build environments
export ONNX_ML=1
export CMAKE_ARGS="-DPYTHON_INCLUDE_DIR=/opt/python/${PY_VER}/include/python$PY_VERSION -DONNX_USE_LITE_PROTO=ON"

# Install Python dependency
$PIP_INSTALL_COMMAND -r requirements-release.txt || { echo "Installing Python requirements failed."; exit 1; }

# Build wheels
if [ "$GITHUB_EVENT_NAME" == "schedule" ]; then
    sed -i 's/name = "onnx"/name = "onnx-weekly"/' 'pyproject.toml'
    ONNX_PREVIEW_BUILD=1 $PYTHON_COMMAND -m build --wheel || { echo "Building wheels failed."; exit 1; }
else
    $PYTHON_COMMAND -m build --wheel || { echo "Building wheels failed."; exit 1; }
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
