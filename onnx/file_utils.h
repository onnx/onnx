/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "onnx/common/status.h"

using namespace ONNX_NAMESPACE::Common;

namespace ONNX_NAMESPACE {
#ifdef _WIN32
Status FileOpenRd(const std::wstring& path, /*out*/ int& fd);

Status FileOpenWr(const std::wstring& path, /*out*/ int& fd);

#endif

Status FileOpenRd(const std::string& path, /*out*/ int& fd);

Status FileOpenWr(const std::string& path, /*out*/ int& fd); 

Status FileClose(int fd);

Status GetFileLength(int fd, /*out*/ size_t& file_size);

} // namespace ONNX_NAMESPACE
