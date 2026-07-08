// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

#include "onnx/defs/shape_inference.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {

template <typename T>
ONNX_API TensorProto ToTensor(const T& value);

template <typename T>
ONNX_API TensorProto ToTensor(const std::vector<T>& values);

template <typename T>
std::vector<T> ParseData(const TensorProto* tensor_proto);

// Return the single element of a tensor that is required to hold exactly one
// value, i.e. a scalar or a single-element rank-1 tensor. The element count is
// validated on the tensor's shape before the element is read, so this never
// reads out of bounds. Fails shape inference if the tensor holds anything other
// than exactly one element.
template <typename T>
T squeezeSingleElementTensor(const TensorProto* tensor_proto) {
  if (tensor_proto->dims_size() > 1) {
    fail_shape_inference("Tensor '", tensor_proto->name(), "' must be a scalar or rank 1 tensor.");
  }
  if (tensor_proto->dims_size() == 1 && tensor_proto->dims(0) != 1) {
    fail_shape_inference("Tensor '", tensor_proto->name(), "' must have exactly one element.");
  }
  return ParseData<T>(tensor_proto)[0];
}

} // namespace ONNX_NAMESPACE
