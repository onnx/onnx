// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE::defs::optional::utils {
// Optional and OptionalGetElement accept separate element and optional type sets
// because historical schemas do not wrap the same element type set.

std::function<void(OpSchema&)> OptionalOpGenerator(
    std::vector<std::string> tensor_and_sequence_types,
    std::vector<std::string> optional_types);

std::function<void(OpSchema&)> OptionalHasElementOpGenerator(std::vector<std::string> o_types);

std::function<void(OpSchema&)> OptionalGetElementOpGenerator(
    std::vector<std::string> optional_types,
    std::vector<std::string> tensor_and_sequence_types);
} // namespace ONNX_NAMESPACE::defs::optional::utils
