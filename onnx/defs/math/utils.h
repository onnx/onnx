/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
namespace defs {
namespace math {
namespace utils {
template <typename T>
T GetScalarValueFromTensor(const ONNX_NAMESPACE::TensorProto* t);

std::function<void(OpSchema&)>
SoftmaxFamilyDocGenerator(const char* name, const char* description, const char* equation);
} // namespace utils
} // namespace math
} // namespace defs
} // namespace ONNX_NAMESPACE
