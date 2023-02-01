#!/usr/bin/env bash

# Test backend test data
git status
# Skip *output_*.pb because NumPy functions might behave differently on different platforms
# Skip test_log's input.pb because it uses np.random, which might behave differently on different platforms
git diff --exit-code -- . ':!onnx/onnx-data.proto' ':!onnx/onnx-data.proto3' ':!*output_*.pb' ':!*input_*.pb'
if [ $? -ne 0 ]; then
echo "git diff for test generation returned failures. Please check updated node test files"
exit 1
fi
git diff --exit-code --diff-filter=ADR -- . ':!onnx/onnx-data.proto' ':!onnx/onnx-data.proto3'
if [ $? -ne 0 ]; then
echo "Test generation returned failures. Please check the number of node test files (input_*.pb or output_*.pb)"
exit 1
fi
