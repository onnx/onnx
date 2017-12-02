#!/bin/bash

set -e

echo "Auto generating docs"
python onnx/defs/gen_doc.py

echo "Auto generating proto files"
python onnx/gen_proto.py

echo "Auto generating backend test data"
backend-test-tools generate-data

echo "Success"
