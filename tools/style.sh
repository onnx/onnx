#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0


set +o errexit
set -o nounset


cd "$(git rev-parse --show-toplevel)"

err=0
trap 'err=1' ERR

echo -e "\n::group:: ===> check flake8..."
flake8 onnx tools workflow_scripts
echo -e "::endgroup::"

echo -e "\n::group:: ===> check mypy"
mypy . --no-site-packages
echo -e "::endgroup::"

echo -e "\n::group:: ===> run clang-format"
git ls-files --exclude-standard -- '*/*.cc' '*/*.h' | \
    xargs ${CLANG_FORMAT_BIN:-clang-format} -i
echo -e "::endgroup::"

git diff --exit-code

test $err = 0 # Return non-zero if any command failed
