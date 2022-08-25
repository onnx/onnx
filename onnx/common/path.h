/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include <string>
#ifdef _WIN32
#include <Windows.h>
#endif

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

#ifdef _WIN32
inline std::wstring utf8str_to_wstring(const std::string& utf8str) {
  int size_required = MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(),
                                          (int)utf8str.size(), NULL, 0);
  std::wstring ws_str(size_required, 0);
  MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), (int)utf8str.size(),
                      &ws_str[0], size_required);
  return ws_str;
}
#endif

} // namespace ONNX_NAMESPACE
