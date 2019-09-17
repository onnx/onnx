// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include "onnx/common/ir.h"

namespace ONNX_NAMESPACE {

template <typename T>
const std::vector<T> ParseData(const Tensor* tensor);

} // namespace ONNX_NAMESPACE
