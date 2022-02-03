#!/bin/bash

# SPDX-License-Identifier: Apache-2.0


set -eu


cd "$(git rev-parse --show-toplevel)"

echo -e "===> check flake8"
flake8 onnx

echo -e "\n===> check mypy"
mypy . --no-site-packages

echo -e "\n===> run clang-format"
git ls-files --exclude-standard -- '*/*.cc' '*/*.h' | \
    xargs ${CLANG_FORMAT_BIN:-clang-format} -i

git diff --exit-code