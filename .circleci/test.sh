#!/bin/bash

set -ex

cd "$PYTORCH_DIR"
exec scripts/onnx/test.sh
