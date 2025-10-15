// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <onnx/defs/schema.h>
#include <onnx/onnx_pb.h>

#include <cstdio>
using namespace ONNX_NAMESPACE;
int main() {
  puts("Link ONNX successfully!");
  std::vector<std::string> all_fixed_size_types;

    std::vector<std::string> all_tensor_types = OpSchema::all_tensor_types_ir9();
    all_fixed_size_types.insert(all_fixed_size_types.end(), all_tensor_types.begin(), all_tensor_types.end());

    ONNX_OPERATOR_SCHEMA(MemcpyToHost)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output", "T")
        .TypeConstraint(
            "T",
            all_fixed_size_types,
            "Constrain to all fixed size tensor and sequence types. If the dtype attribute is not provided this must be a valid output type.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .SetDoc(R"DOC(
Internal copy node
)DOC");

  return 0;
}
