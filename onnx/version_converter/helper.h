// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Helper Methods for Adapters

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "onnx/common/assertions.h"
#include "onnx/common/ir.h"
#include "onnx/defs/tensor_util.h"

namespace ONNX_NAMESPACE::version_conversion {
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
      "expected INT64 tensor, got elem_type=",
      tensor.elem_type())
  if (tensor.is_raw_data()) {
    const size_t raw_bytes = tensor.raw().size();
    // elem_num() returns 1 for scalars, so covers dims=[].
    ONNX_ASSERTM(
        raw_bytes == static_cast<size_t>(tensor.elem_num()) * sizeof(int64_t),
        "INT64 tensor: ",
        raw_bytes,
        " raw bytes does not match dims (",
        tensor.elem_num(),
        " elements)")
  }
  return ParseData<int64_t>(&tensor);
}

// Puts `to` in place of `from`: `to` takes over the name of `from`, so graph outputs and
// values captured by subgraphs (bound by name) now refer to it, and uses of `from` in the
// current graph are rewired to `to`. `from` gets a fresh name.
inline void ReplaceValueKeepingName(Graph& graph, Value* from, Value* to) {
  const std::string name = from->uniqueName();
  from->setUniqueName(graph.getNextUniqueName(), false);
  to->setUniqueName(name, false);
  // With the name moved, uses() only reports users in the current graph.
  for (Use u : from->uses()) {
    u.user->replaceInputWith(from, to);
  }
}
} // namespace ONNX_NAMESPACE::version_conversion
