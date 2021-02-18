/*
 * SPDX-License-Identifier: Apache-2.0
 */

//#include <google/protobuf/io/coded_stream.h>
//#include <fcntl.h>
//#include <io.h>
#include "onnx/onnx_pb.h"
#include "onnx/common/status.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "onnx/file_utils.h"
#include "onnx/common/common.h"
#include "onnx/string_utils.h"

using namespace onnx::Common;

namespace ONNX_NAMESPACE {

static constexpr int DEFAULT_PROTOBUF_BLOCK_SIZE = 4 * 1024 * 1024;

Status LoadModel(const std::string& file_path, ModelProto& model_proto) {
  int fd;
  Status status = FileOpenRd(file_path, fd);

  if (fd < 0 || !status.IsOK()) {
    return Status(
        StatusCategory::SYSTEM,
        StatusCode::INVALID_ARGUMENT,
        "model_load_utils: Could not open file for reading." + file_path);
  }

#if GOOGLE_PROTOBUF_VERSION >= 3002000
  size_t file_size = 0;
  int block_size = -1;
  Status st = GetFileLength(fd, file_size);
  if (st.IsOK()) {
    block_size = std::min(DEFAULT_PROTOBUF_BLOCK_SIZE, static_cast<int>(file_size));
  }
  ::google::protobuf::io::FileInputStream input(fd, block_size);
  const bool result = model_proto.ParseFromZeroCopyStream(&input) && input.GetErrno() == 0;

#else
  // CNTK uses ORT as a submodule in order to use its GraphIR code.
  // CNTK needs to be built with protobuf 3.1.0 for its version specific features.
  // This code block is needed to support CNTK and any other
  // GraphIR client that will be built with protobuf at a version older than 3.2.0.
  FileInputStream fs(fd);
  CodedInputStream cis(&fs);

  // Allows protobuf library versions < 3.2.0 to parse messages greater than 64MB.
  cis.SetTotalBytesLimit(INT_MAX);
  const bool result = model_proto.ParseFromCodedStream(&cis);

#endif

  if (!result) {
    FileClose(fd);
    return Status(StatusCategory::ONNX, INVALID_PROTOBUF, "Protobuf parsing failed.");
  }

  return FileClose(fd);
}

Status SaveModel(const std::string& file_path, ModelProto& model_proto) {
  int p_fd;
  Status status = FileOpenWr(file_path, p_fd);

  if (p_fd < 0 || !status.IsOK()) {
    return Status(StatusCategory::SYSTEM, StatusCode::INVALID_ARGUMENT, "model_load_utils: Could not open file for writing." + file_path);
  }

  ::google::protobuf::io::FileOutputStream output(p_fd);
  if (model_proto.SerializeToZeroCopyStream(&output) && output.Flush()) {
    return FileClose(p_fd);
  }

  FileClose(p_fd);
  return Status(
      StatusCategory::ONNX,
      StatusCode::INVALID_PROTOBUF,
      "model_load_utils: INVALID_PROTOBUF Protobuf serialization failed.");
}

} // namespace ONNX_NAMESPACE
