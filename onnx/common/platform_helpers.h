// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

namespace ONNX_NAMESPACE {

// Determine if the processor is little endian or not
inline bool is_processor_little_endian() {
  int num = 1;
  if (*(char*)&num == 1)
    return true;
  return false;
}

} // namespace ONNX_NAMESPACE
