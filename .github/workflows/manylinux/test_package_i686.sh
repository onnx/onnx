#!/bin/bash

set -e -x

# CLI arguments
PY_VERSION=$1

# Need to be updated if there is a new Python Version
declare -A python_map=( ["3.6"]="cp36-cp36m" ["3.7"]="cp37-cp37m" ["3.8"]="cp38-cp38" ["3.9"]="cp39-cp39")
PY_VER=${python_map[$PY_VERSION]}

yum install -y protobuf-devel cmake3

PIP_COMMAND="/opt/python/${PY_VER}/bin/pip install --no-cache-dir"
PYTHON_COMAND="/opt/python/"${PY_VER}"/bin/python"
PYTEST_COMAND="/opt/python/${PY_VER}/bin/pytest"

$PIP_COMMAND --upgrade pip
$PIP_COMMAND numpy protobuf==3.11.3
$PIP_COMMAND dist/*-manylinux2010_i686.whl

# pytest with the built wheel
# TODO Remove fixed ipython 7.16.1 once ONNX has removed Python 3.6
$PIP_COMMAND pytest==5.4.3 nbval ipython==7.16.1
$PYTEST_COMAND

# Test generated backend test data
$PYTHON_COMAND onnx/backend/test/cmd_tools.py generate-data
# Only test generated backend node test data
$PYTEST_COMAND onnx/test/test_backend_test.py -k OnnxBackendNodeModelTest

echo "Succesfully test the wheel"