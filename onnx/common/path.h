// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <string>

#ifdef _WIN32
// windows.h has preproc definitions for min and max, which prevents from using std::min and std::max.
//  defining NOMINMAX disables the preproc macro.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "onnx/checker.h"
#endif

namespace ONNX_NAMESPACE {

#ifdef _WIN32
inline std::wstring utf8str_to_wstring(const std::string& utf8str) {
  if (utf8str.empty()) {
    return std::wstring();
  }
  int len = static_cast<int>(utf8str.size());
  auto size_required =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS | MB_PRECOMPOSED, utf8str.data(), len, nullptr, 0);
  if (size_required == 0) {
    auto last_error = GetLastError();
    fail_check("MultiByteToWideChar in utf8str_to_wstring returned error:", last_error);
  }
  std::wstring ws_str(size_required, 0);
  auto converted_size = MultiByteToWideChar(
      CP_UTF8, MB_ERR_INVALID_CHARS | MB_PRECOMPOSED, utf8str.data(), len, &ws_str[0], size_required);
  if (converted_size == 0) {
    auto last_error = GetLastError();
    fail_check("MultiByteToWideChar in utf8str_to_wstring returned error:", last_error);
  }
  return ws_str;
}

inline std::string wstring_to_utf8str(const std::wstring& ws_str) {
  if (ws_str.empty()) {
    return std::string();
  }
  int len = static_cast<int>(ws_str.size());
  auto size_required =
      WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, ws_str.data(), len, nullptr, 0, nullptr, nullptr);
  if (size_required == 0) {
    auto last_error = GetLastError();
    fail_check("WideCharToMultiByte in wstring_to_utf8str returned error:", last_error);
  }
  std::string utf8str(size_required, 0);
  auto converted_size = WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, ws_str.data(), len, &utf8str[0], size_required, nullptr, nullptr);
  if (converted_size == 0) {
    auto last_error = GetLastError();
    fail_check("WideCharToMultiByte in wstring_to_utf8str returned error:", last_error);
  }
  return utf8str;
}

#endif

// Construct a std::filesystem::path from a UTF-8 string.
// On Windows, converts via wide-char so the result is independent of the
// active code page.  On POSIX, UTF-8 is the native encoding.
inline std::filesystem::path utf8_to_path(const std::string& utf8) {
#ifdef _WIN32
  return std::filesystem::path(utf8str_to_wstring(utf8));
#else
  return std::filesystem::path(utf8);
#endif
}

// Return the UTF-8 representation of a filesystem path.
inline std::string path_to_utf8(const std::filesystem::path& p) {
#ifdef _WIN32
  return wstring_to_utf8str(p.wstring());
#else
  return p.string();
#endif
}

} // namespace ONNX_NAMESPACE
