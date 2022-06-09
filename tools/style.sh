#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0


set -o errexit
set -o nounset


cd "$(git rev-parse --show-toplevel)"

echo -e "===> check flake8"
flake8 onnx tools workflow_scripts

echo -e "\n===> check mypy"
mypy . --no-site-packages

echo -e "\n===> run clang-format"
git ls-files --exclude-standard -- '*/*.cc' '*/*.h' | \
    xargs ${CLANG_FORMAT_BIN:-clang-format} -i

git diff --exit-code
