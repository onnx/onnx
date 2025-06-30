/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {

template <typename T>
TensorProto ToTensor(const T& value);

template <typename T>
TensorProto ToTensor(const std::vector<T>& values);

template <typename T>
std::vector<T> ParseData(const TensorProto* tensor_proto);

} // namespace ONNX_NAMESPACE
