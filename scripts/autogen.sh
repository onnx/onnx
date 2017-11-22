#!/bin/bash

set -e

if ! python -c 'import onnx' 2>/dev/null; then
    echo "Error: You need to install onnx first"
    exit 1
fi

echo "Auto generating docs"
python onnx/defs/gen_doc.py

echo "Auto generating proto files"
python onnx/gen_proto.py

echo "Auto generating backend test data"
backend-test-tools generate-data

echo "Success"
