/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <sys/stat.h>
#include <limits>

#include "onnx/common/status.h"
#include "onnx/onnx_pb.h"

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif // _WIN32

using namespace ONNX_NAMESPACE::Common;
namespace ONNX_NAMESPACE {
#ifdef _WIN32
Status FileOpenRd(const std::wstring& path, /*out*/ int& fd) {
  _wsopen_s(&fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
  if (0 > fd) {
    return Status(Common::SYSTEM, StatusCode::FAIL);
  }
  return Status::OK();
}

Status FileOpenWr(const std::wstring& path, /*out*/ int& fd) {
  _wsopen_s(
      &fd, path.c_str(), _O_CREAT | _O_TRUNC | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
  if (0 > fd) {
    return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
  }
  return Status::OK();
}
#endif

Status FileOpenRd(const std::string& path, /*out*/ int& fd) {
#ifdef _WIN32
  _sopen_s(&fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#else
  fd = open(path.c_str(), O_RDONLY);
#endif
  if (0 > fd) {
    return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
  }
  return Status::OK();
}

Status FileOpenWr(const std::string& path, /*out*/ int& fd) {
#ifdef _WIN32
  _sopen_s(
      &fd, path.c_str(), _O_CREAT | _O_TRUNC | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
#else
  fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
#endif

  if (0 > fd) {
    return Status(Common::SYSTEM, StatusCode::FAIL);
  }
  return Status::OK();
}

Status FileClose(int fd) {
#ifdef _WIN32
  int ret = _close(fd);
#else
  int ret = close(fd);
#endif
  if (0 != ret) {
    return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
  }
  return Status::OK();
}

Status GetFileLength(int fd, /*out*/ size_t& file_size) {
  if (fd < 0) {
    return Status(StatusCategory::ONNX, StatusCode::INVALID_ARGUMENT, "Invalid fd was supplied");
  }

#ifdef _WIN32
  struct _stat buf;
  int rc = _fstat(fd, &buf);
#else
  struct stat buf;
  int rc = fstat(fd, &buf);
#endif
  if (rc < 0) {
    return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
  }

  if (buf.st_size < 0) {
    return Status(StatusCategory::SYSTEM, StatusCode::FAIL, "Received negative size from stat call");
  }

  if (static_cast<unsigned long long>(buf.st_size) > std::numeric_limits<size_t>::max()) {
    return Status(StatusCategory::SYSTEM, StatusCode::FAIL, "File is too large.");
  }

  file_size = static_cast<size_t>(buf.st_size);
  return Status::OK();
}
} // namespace ONNX_NAMESPACE
