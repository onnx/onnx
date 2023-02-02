#!/usr/bin/env bash

# Check auto-gen files up-to-date.
git status
if ! git diff --exit-code -- . ':(exclude)onnx/onnx-data.proto' ':(exclude)onnx/onnx-data.proto3'; then
    echo "git diff returned failures"
    exit 1
fi
