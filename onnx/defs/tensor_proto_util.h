// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#pragma once

#include "onnx/onnx-operators_pb.h"

namespace ONNX_NAMESPACE {

template <typename T>
TensorProto ToTensor(const T& value);

template <typename T>
TensorProto ToTensor(const std::vector<T>& values);


} // namespace ONNX_NAMESPACE
