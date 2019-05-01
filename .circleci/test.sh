#!/bin/bash

set -ex

source /tmp/venv/bin/activate
cd /tmp/pytorch
CI=1 exec "scripts/onnx/test.sh"
