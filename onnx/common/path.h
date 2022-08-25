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

template <typename CHAR>
bool is_path_separator(CHAR c) {
  // Windows accept / as path separator.
  if (k_preferred_path_separator == "\\") {
    return c == '\\' || c == '/';
  }

  return c == k_preferred_path_separator[0];
}

template <typename STRING>
void normalize_separator(STRING& path) {
  char preferred_sep = k_preferred_path_separator[0];
  if (preferred_sep == '/') {
    // Do nothing on linux.
    return;
  }

  for (size_t i = 0; i < path.size(); i++) {
    if (is_path_separator(path[i]) && path[i] != preferred_sep) {
      path[i] = preferred_sep;
    }
  }
}

std::string path_join(const std::string& origin, const std::string& append);
std::string clean_relative_path(const std::string& path);
std::wstring clean_relative_path(const std::wstring& path);

} // namespace ONNX_NAMESPACE
