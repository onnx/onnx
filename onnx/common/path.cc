/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#include "onnx/common/path.h"

namespace ONNX_NAMESPACE {

#define PATH_JOIN(string_type, separator)                                         \
  template <>                                                                     \
  string_type path_join(const string_type& origin, const string_type& append) {   \
    if (origin.find_last_of(separator) != origin.length() - separator.length()) { \
      return origin + separator + append;                                         \
    }                                                                             \
    return origin + append;                                                       \
  }

#define CLEAN_RELATIVE_PATH(string_type, char_type, separator, dot, input_type)                               \
  template <>                                                                                                 \
  string_type clean_relative_path(const input_type input_path) {                                              \
    string_type path = string_type(input_path);                                                               \
    if (path.empty()) {                                                                                       \
      return dot;                                                                                             \
    }                                                                                                         \
    string_type out;                                                                                          \
    char_type sep = separator[0];                                                                             \
    char_type dot_char = dot[0];                                                                              \
    size_t n = path.size();                                                                                   \
    size_t r = 0;                                                                                             \
    size_t dotdot = 0;                                                                                        \
    while (r < n) {                                                                                           \
      if (is_path_separator(path[r])) {                                                                       \
        r++;                                                                                                  \
        continue;                                                                                             \
      }                                                                                                       \
      if (path[r] == dot_char && (r + 1 == n || is_path_separator(path[r + 1]))) {                            \
        r++;                                                                                                  \
        continue;                                                                                             \
      }                                                                                                       \
      if (path[r] == dot_char && path[r + 1] == dot_char && (r + 2 == n || is_path_separator(path[r + 2]))) { \
        r += 2;                                                                                               \
        if (out.size() > dotdot) {                                                                            \
          while (out.size() > dotdot && !is_path_separator(out.back())) {                                     \
            out.pop_back();                                                                                   \
          }                                                                                                   \
          if (!out.empty())                                                                                   \
            out.pop_back();                                                                                   \
        } else {                                                                                              \
          if (!out.empty()) {                                                                                 \
            out.push_back(sep);                                                                               \
          }                                                                                                   \
          out.push_back(dot_char);                                                                            \
          out.push_back(dot_char);                                                                            \
          dotdot = out.size();                                                                                \
        }                                                                                                     \
        continue;                                                                                             \
      }                                                                                                       \
                                                                                                              \
      if (!out.empty() && out.back() != sep) {                                                                \
        out.push_back(sep);                                                                                   \
      }                                                                                                       \
                                                                                                              \
      for (; r < n && !is_path_separator(path[r]); r++) {                                                     \
        out.push_back(path[r]);                                                                               \
      }                                                                                                       \
    }                                                                                                         \
                                                                                                              \
    if (out.empty()) {                                                                                        \
      out.push_back(dot_char);                                                                                \
    }                                                                                                         \
                                                                                                              \
    normalize_separator(out);                                                                                 \
                                                                                                              \
    return out;                                                                                               \
  }

PATH_JOIN(std::string, k_preferred_path_separator);
CLEAN_RELATIVE_PATH(std::string, char, k_preferred_path_separator, ".", std::string&);
CLEAN_RELATIVE_PATH(std::string, char, k_preferred_path_separator, ".", char*);

#ifdef _WIN32
PATH_JOIN(std::wstring, w_k_preferred_path_separator);
CLEAN_RELATIVE_PATH(std::wstring, wchar_t, w_k_preferred_path_separator, L".", std::wstring&);
CLEAN_RELATIVE_PATH(std::wstring, wchar_t, w_k_preferred_path_separator, L".", wchar_t*);

#endif

} // namespace ONNX_NAMESPACE
