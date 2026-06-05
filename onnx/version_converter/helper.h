// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Helper Methods for Adapters

#pragma once

#include <cstdint>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/common/ir.h"
#include "onnx/defs/tensor_util.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {
int check_numpy_unibroadcastable_and_require_broadcast(
    const std::vector<Dimension>& input1_sizes,
    const std::vector<Dimension>& input2_sizes);

void assert_numpy_multibroadcastable(
    const std::vector<Dimension>& input1_sizes,
    const std::vector<Dimension>& input2_sizes);

void assertNotParams(const std::vector<Dimension>& sizes);

void assertInputsAvailable(const ArrayRef<Value*>& inputs, const char* name, uint64_t num_inputs);

// Decode an INT64 tensor; rejects mismatched element type or dims/raw byte length.
inline std::vector<int64_t> ReadInt64Tensor(const Tensor& tensor) {
  ONNX_ASSERTM(
      tensor.elem_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64,
      "expected INT64 tensor, got elem_type=%d",
      tensor.elem_type())
  if (tensor.is_raw_data()) {
    const size_t raw_bytes = tensor.raw().size();
    // elem_num() returns 1 for scalars, so covers dims=[].
    ONNX_ASSERTM(
        raw_bytes == static_cast<size_t>(tensor.elem_num()) * sizeof(int64_t),
        "INT64 tensor: %zu raw bytes does not match dims (%lld elements)",
        raw_bytes,
        static_cast<long long>(tensor.elem_num()))
  }
  return ParseData<int64_t>(&tensor);
}
} // namespace version_conversion
} // namespace ONNX_NAMESPACE
