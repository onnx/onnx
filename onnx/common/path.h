/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Copyright (c) ONNX Project Contributors.

#pragma once

#ifdef _WIN32
#include <string>
// windows.h has preproc definitions for min and max, which prevents from using std::min and std::max.
//  defining NOMINMAX disables the preproc macro.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace ONNX_NAMESPACE {

inline std::wstring utf8str_to_wstring(const std::string& utf8str) {
  if (utf8str.size() > INT_MAX) {
    fail_check("utf8str_to_wstring: string is too long for converting to wstring.");
  }
  int size_required = MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), static_cast<int>(utf8str.size()), nullptr, 0);
  std::wstring ws_str(size_required, 0);
  MultiByteToWideChar(CP_UTF8, 0, utf8str.c_str(), static_cast<int>(utf8str.size()), &ws_str[0], size_required);
  return ws_str;
}
inline std::string wstring_to_utf8str(const std::wstring& ws_str) {
  if (ws_str.size() > INT_MAX) {
    fail_check("wstring_to_utf8str: string is too long for converting to UTF-8.");
  }
  int size_required =
      WideCharToMultiByte(CP_UTF8, 0, ws_str.c_str(), static_cast<int>(ws_str.size()), nullptr, 0, nullptr, nullptr);
  std::string utf8str(size_required, 0);
  WideCharToMultiByte(
      CP_UTF8, 0, ws_str.c_str(), static_cast<int>(ws_str.size()), &utf8str[0], size_required, nullptr, nullptr);
  return utf8str;
}

} // namespace ONNX_NAMESPACE
#endif
