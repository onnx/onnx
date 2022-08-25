/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include <string>

namespace ONNX_NAMESPACE {

#ifdef _WIN32
const std::string k_preferred_path_separator = "\\";
#else // POSIX
const std::string k_preferred_path_separator = "/";
#endif

std::string path_join(const std::string& origin, const std::string& append);
void normalize_separator(const std::string& path);
void normalize_separator(const std::wstring& path);
std::string clean_relative_path(const std::string& path);
std::wstring clean_relative_path(const std::wstring& path);

} // namespace ONNX_NAMESPACE
