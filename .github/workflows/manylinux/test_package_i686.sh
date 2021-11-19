#!/bin/bash

set -e -x

# CLI arguments
PY_VERSION=$1

# Need to be updated if there is a new Python Version
declare -A python_map=( ["3.6"]="cp36-cp36m" ["3.7"]="cp37-cp37m" ["3.8"]="cp38-cp38" ["3.9"]="cp39-cp39")
PY_VER=${python_map[$PY_VERSION]}

yum install -y protobuf-devel cmake3

PYTHON_BIN="/opt/python/${PY_VER}/bin/"
PIP_INSTALL_COMMAND="${PYTHON_BIN}pip install --no-cache-dir -q"
PIP_UNINSTALL_COMMAND="${PYTHON_BIN}pip uninstall -y"
PYTHON_COMAND="${PYTHON_BIN}python"
PYTEST_COMMAND="${PYTHON_BIN}pytest"

$PIP_INSTALL_COMMAND --upgrade pip

# pip install -r requirements-release will bump into issue in i686 due to pip install cryptography failure
$PIP_INSTALL_COMMAND numpy protobuf==3.16.0 pytest==5.4.3 nbval ipython==7.16.1 || { echo "Installing Python requirements failed."; exit 1; }
$PIP_INSTALL_COMMAND dist/*manylinux2010_i686.whl

# pytest with the built wheel
$PYTEST_COMMAND

# Test generated backend test data
$PYTHON_COMMAND onnx/backend/test/cmd_tools.py generate-data
# Only test generated backend node test data
$PYTEST_COMMAND onnx/test/test_backend_test.py -k OnnxBackendNodeModelTest

# Verify ONNX with the latest numpy
$PIP_UNINSTALL_COMMAND numpy onnx && $PIP_INSTALL_COMMAND numpy
$PIP_INSTALL_COMMAND dist/*manylinux2010_i686.whl
$PYTEST_COMMAND

# Verify ONNX with the latest protobuf
$PIP_UNINSTALL_COMMAND protobuf onnx && $PIP_INSTALL_COMMAND protobuf
$PIP_INSTALL_COMMAND dist/*manylinux2010_i686.whl
$PYTEST_COMMAND

# Verify ONNX the minimum supported protobuf (from requirements.txt)
$PIP_UNINSTALL_COMMAND protobuf onnx && $PIP_INSTALL_COMMAND protobuf==3.12.2
$PIP_INSTALL_COMMAND dist/*manylinux2010_i686.whl
$PYTEST_COMMAND

echo "Succesfully test the wheel"