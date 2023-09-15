/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
template <typename T>
T GetScalarValueFromTensor(const TensorProto* t);
} // namespace ONNX_NAMESPACE
