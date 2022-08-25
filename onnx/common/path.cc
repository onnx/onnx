/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/common/path.h"

namespace ONNX_NAMESPACE {

std::string path_join(const std::string& origin, const std::string& append) {
  if (origin.find_last_of(k_preferred_path_separator) != origin.length() - k_preferred_path_separator.length()) {
    return origin + k_preferred_path_separator + append;
  }
  return origin + append;
}

std::string clean_relative_path(const std::string& path) {
  if (path.empty()) {
    return ".";
  }

  std::string out;

  char sep = k_preferred_path_separator[0];
  size_t n = path.size();

  size_t r = 0;
  size_t dotdot = 0;

  while (r < n) {
    if (is_path_separator(path[r])) {
      r++;
      continue;
    }

    if (path[r] == '.' && (r + 1 == n || is_path_separator(path[r + 1]))) {
      r++;
      continue;
    }

    if (path[r] == '.' && path[r + 1] == '.' && (r + 2 == n || is_path_separator(path[r + 2]))) {
      r += 2;

      if (out.size() > dotdot) {
        while (out.size() > dotdot && !is_path_separator(out.back())) {
          out.pop_back();
        }
        if (!out.empty())
          out.pop_back();
      } else {
        if (!out.empty()) {
          out.push_back(sep);
        }

        out.push_back('.');
        out.push_back('.');
        dotdot = out.size();
      }

      continue;
    }

    if (!out.empty() && out.back() != sep) {
      out.push_back(sep);
    }

    for (; r < n && !is_path_separator(path[r]); r++) {
      out.push_back(path[r]);
    }
  }

  if (out.empty()) {
    out.push_back('.');
  }

  // Use 1 separator in path.
  normalize_separator(out);

  return out;
}

std::wstring clean_relative_path(const std::wstring& path) {
  if (path.empty()) {
    return L".";
  }

  std::wstring out;

  wchar_t sep = L'\\';
  size_t n = path.size();

  size_t r = 0;
  size_t dotdot = 0;

  while (r < n) {
    if (is_path_separator(path[r])) {
      r++;
      continue;
    }

    if (path[r] == L'.' && (r + 1 == n || is_path_separator(path[r + 1]))) {
      r++;
      continue;
    }

    if (path[r] == L'.' && path[r + 1] == L'.' && (r + 2 == n || is_path_separator(path[r + 2]))) {
      r += 2;

      if (out.size() > dotdot) {
        while (out.size() > dotdot && !is_path_separator(out.back())) {
          out.pop_back();
        }
        if (!out.empty())
          out.pop_back();
      } else {
        if (!out.empty()) {
          out.push_back(sep);
        }

        out.push_back(L'.');
        out.push_back(L'.');
        dotdot = out.size();
      }

      continue;
    }

    if (!out.empty() && out.back() != sep) {
      out.push_back(sep);
    }

    for (; r < n && !is_path_separator(path[r]); r++) {
      out.push_back(path[r]);
    }
  }

  if (out.empty()) {
    out.push_back(L'.');
  }

  // Use 1 separator in path.
  normalize_separator(out);

  return out;
}

} // namespace ONNX_NAMESPACE
