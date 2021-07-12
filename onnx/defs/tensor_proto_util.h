/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/onnx-operators_pb.h"

namespace ONNX_NAMESPACE {

template <typename T>
TensorProto ToTensor(const T& value);

template <typename T>
TensorProto ToTensor(const std::vector<T>& values);

template <typename T>
inline TensorProto To1DTensor(const std::vector<T>& values) {
  TensorProto result = ToTensor<T>(values);
  result.add_dims(values.size());
  return result;
}

template <typename T>
const std::vector<T> ParseData(const TensorProto* tensor_proto);

} // namespace ONNX_NAMESPACE
