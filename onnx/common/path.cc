/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/common/path.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {

const std::string path_join(const std::string& origin, const std::string& append) {
    std::string new_path = origin;
    if (new_path.length() >= k_preferred_path_separator.length() && 
        new_path.substr(new_path.length () - k_preferred_path_separator.length(), k_preferred_path_separator.length()) != k_preferred_path_separator) {
        new_path += k_preferred_path_separator;
    }
    return new_path + append;
}

} // namespace ONNX_NAMESPACE
