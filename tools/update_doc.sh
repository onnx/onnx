#!/bin/bash

# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0


# export ONNX_ML=0, # if you need to disable ONNX_ML

python_exist=`command -v python`
if [ -z $python_exist ]; then
  echo "No python is found, please set it in your environment."
  exit 1
fi

if [ ! -f "onnx/defs/gen_doc.py" ]; then
  echo "Please run this script in the ONNX root folder."
  exit 1
fi

set -e

echo -e "===> recompile onnx"
python setup.py develop --user

echo -e "\n===> regenerate test data from node test"
python onnx/backend/test/cmd_tools.py generate-data

echo -e "\n===> regenerate stats of test data"
python onnx/backend/test/stat_coverage.py

echo -e "\n===> regenerate the docs"
python onnx/defs/gen_doc.py

echo -e "\n===> the update is done!"
