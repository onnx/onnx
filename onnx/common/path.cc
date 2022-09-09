/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/common/path.h"

namespace ONNX_NAMESPACE {
#ifdef _WIN32
#else

std::string path_join(const std::string& origin, const std::string& append) {
  if (origin.find_last_of(k_preferred_path_separator) != origin.length() - 1) {
    return origin + std::string(k_preferred_path_separator) + append;
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
    if (path[r] == sep) {
      r++;
      continue;
    }

    if (path[r] == '.' && (r + 1 == n || path[r + 1] == sep)) {
      r++;
      continue;
    }

    if (path[r] == '.' && path[r + 1] == '.' && (r + 2 == n || path[r + 2] == sep)) {
      r += 2;

      if (out.size() > dotdot) {
        while (out.size() > dotdot && !out.back() == sep) {
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

    for (; r < n && !path[r] == sep; r++) {
      out.push_back(path[r]);
    }
  }

  if (out.empty()) {
    out.push_back('.');
  }

  return out;
}
#endif

} // namespace ONNX_NAMESPACE
