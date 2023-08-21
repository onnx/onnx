/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
namespace defs::math::utils {
template <typename T>
static T GetScalarValueFromTensor(const ONNX_NAMESPACE::TensorProto* t);

std::function<void(OpSchema&)>
SoftmaxFamilyDocGenerator(const char* name, const char* description, const char* equation);
} // namespace defs::math::utils
} // namespace ONNX_NAMESPACE
