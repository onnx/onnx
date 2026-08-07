// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
namespace defs {
namespace optional {
namespace utils {
// Using types::Optional(tensor_and_sequence_types) may not equal to optional_types.
// See OpSchema::all_optional_types_ir13().
// This is the reason Optional and OptionalGetElement generator takes both optional and element data type arrays.

std::function<void(OpSchema&)> OptionalOpGenerator(
    std::vector<std::string> tensor_and_sequence_types,
    std::vector<std::string> optional_types);

std::function<void(OpSchema&)> OptionalHasElementOpGenerator(std::vector<std::string> o_types, bool is_opset18 = false);

std::function<void(OpSchema&)> OptionalGetElementOpGenerator(
    std::vector<std::string> optional_types,
    std::vector<std::string> tensor_and_sequence_types,
    bool is_opset18 = false);
} // namespace utils
} // namespace optional
} // namespace defs
} // namespace ONNX_NAMESPACE
