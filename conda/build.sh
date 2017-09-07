#!/bin/bash
set -e
export ONNX_BINARY_BUILD=1
# Recommended by https://github.com/conda/conda-build/issues/1191
python setup.py install --single-version-externally-managed --record=record.txt
