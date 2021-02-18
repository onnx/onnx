/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fcntl.h>
#include <sys/stat.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "onnx/common/status.h"
#include "onnx/onnx_pb.h"

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h> 
#endif // _WIN32

namespace ONNX_NAMESPACE {

using namespace onnx::Common;
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

Status FileOpenRd(const std::string& path, /*out*/ int& fd) {
  _sopen_s(&fd, path.c_str(), _O_RDONLY | _O_SEQUENTIAL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
  if (0 > fd) {
    return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
  }
  return Status::OK();
}

Status FileOpenWr(const std::string& path, /*out*/ int& fd) {
  _sopen_s(
      &fd, path.c_str(), _O_CREAT | _O_TRUNC | _O_SEQUENTIAL | _O_BINARY | _O_WRONLY, _SH_DENYWR, _S_IREAD | _S_IWRITE);
  if (0 > fd) {
    return Status(Common::SYSTEM, StatusCode::FAIL);
  }
  return Status::OK();
}

Status FileClose(int fd) {
  int ret = _close(fd);
  if (0 != ret) {
    return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
  }
  return Status::OK();
}

Status GetFileLength(int fd, /*out*/ size_t& file_size) {
  if (fd < 0) {
    return Status(StatusCategory::ONNX, StatusCode::INVALID_ARGUMENT, "Invalid fd was supplied");
  }

  struct _stat buf;
  int rc = _fstat(fd, &buf);
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

#else
Status FileOpenRd(const std::string& path, /*out*/ int& fd) {
  fd = open(path.c_str(), O_RDONLY);
  if (0 > fd) {
    return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
  }
  return Status::OK();
}

Status FileOpenWr(const std::string& path, /*out*/ int& fd) {
  fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (0 > fd) {
    return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
  }
  return Status::OK();
}

Status FileClose(int fd) {
  int ret = close(fd);
  if (0 != ret) {
    return Status(StatusCategory::SYSTEM, StatusCode::FAIL);
  }
  return Status::OK();
}

Status GetFileLength(int fd, /*out*/ size_t& file_size) {
  if (fd < 0) {
    return Status(StatusCategory::ONNX, StatusCode::INVALID_ARGUMENT, "Invalid fd was supplied");
  }

  struct stat buf;
  int rc = fstat(fd, &buf);
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

#endif

} // namespace ONNX_NAMESPACE
