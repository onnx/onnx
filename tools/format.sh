#!/bin/bash

# SPDX-License-Identifier: Apache-2.0


set -eu


cd "$(git rev-parse --show-toplevel)"

echo -e "===> check flake8"
flake8 onnx

echo -e "\n===> check mypy"
mypy . --no-site-packages

# Currently, clang-format is not checked on CIs.
if [ "${ENABLE_CLANG_FORMAT}" == "1" ]; then
    echo -e "\n===> run clang-format"
    git ls-files --exclude-standard -- '*/*.cc' '*/*.h' | \
        xargs ${CLANG_FORMAT_BIN:-clang-format} -i
fi

git diff --exit-code