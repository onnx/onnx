/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/common/path.h"

namespace ONNX_NAMESPACE {

std::string path_join(std::string origin, std::string append) {
    if (origin.find_last_of(k_preferred_path_separator) == origin.length() - k_preferred_path_separator.length()) {
        origin += k_preferred_path_separator;
    }
    return origin + append;
}

} // namespace ONNX_NAMESPACE
