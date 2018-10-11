#!/bin/bash

set -ex

exec "$PYTORCH_DIR/scripts/onnx/test.sh"
