#!/bin/bash

set -e -x

# CLI arguments
PY_VERSION=$1

# Need to be updated if there is a new Python Version
declare -A python_map=( ["3.6"]="cp36-cp36m" ["3.7"]="cp37-cp37m" ["3.8"]="cp38-cp38" ["3.9"]="cp39-cp39")
PY_VER=${python_map[$PY_VERSION]}

yum install -y protobuf-devel cmake3

PYTHON_BIN="/opt/python/${PY_VER}/bin/"
PIP_INTALL_COMMAND="${PYTHON_BIN}pip install --no-cache-dir"
PIP_UNINTALL_COMMAND="${PYTHON_BIN}pip uninstall -y"
PYTHON_COMMAND="${PYTHON_BIN}python"
PYTEST_COMMAND="${PYTHON_BIN}pytest"

$PIP_INTALL_COMMAND --upgrade pip
$PIP_INTALL_COMMAND numpy protobuf==3.11.3
$PIP_INTALL_COMMAND dist/*-manylinux2010_i686.whl

# pytest with the built wheel
# TODO Remove fixed ipython 7.16.1 once ONNX has removed Python 3.6
$PIP_INTALL_COMMAND pytest==5.4.3 nbval ipython==7.16.1
$PYTEST_COMMAND

# Test generated backend test data
$PYTHON_COMMAND onnx/backend/test/cmd_tools.py generate-data
# Only test generated backend node test data
$PYTEST_COMMAND onnx/test/test_backend_test.py -k OnnxBackendNodeModelTest

# Verify ONNX with the latest numpy
$PIP_UNINTALL_COMMAND numpy onnx && $PIP_INTALL_COMMAND numpy
$PIP_INTALL_COMMAND dist/*-manylinux2010_i686.whl
$PYTEST_COMMAND

# Verify ONNX with the latest protobuf
$PIP_UNINTALL_COMMAND protobuf onnx && $PIP_INTALL_COMMAND protobuf
$PIP_INTALL_COMMAND dist/*-manylinux2010_i686.whl
$PYTEST_COMMAND

echo "Succesfully test the wheel"