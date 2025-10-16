// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <onnx/defs/schema.h>
#include <onnx/onnx_pb.h>

#include <cstdio>
#include <string>
int main() {
  puts("Link ONNX successfully!");

  auto all_tensor_types = ONNX_NAMESPACE::OpSchema::all_tensor_types_ir9();

  // Test ONNX_OPERATOR_SCHEMA macro, which is borrowed from onnxruntime
  ONNX_OPERATOR_SCHEMA(MemcpyToHost)
      .Input(0, "X", "input", "T")
      .Output(0, "Y", "output", "T")
      .TypeConstraint(
          "T",
          all_tensor_types,
          "Constrain to all fixed size tensor and sequence types. If the dtype attribute is not provided this must be a valid output type.")
      .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput)
      .SetDoc(R"DOC(
Internal copy node
)DOC");

  return 0;
}
