/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Helper Methods for Adapters

#pragma once

#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {
bool HasAttribute(NodeProto* node_proto, const std::string& name);

template <typename T>
std::vector<T> protobuf_repeated_fields_to_vector(const google::protobuf::RepeatedPtrField<T>& repeated);

int check_numpy_unibroadcastable_and_require_broadcast(
    const std::vector<DimensionIR>& input1_sizes,
    const std::vector<DimensionIR>& input2_sizes);
int check_numpy_unibroadcastable_and_require_broadcast(
    std::vector<TensorShapeProto_Dimension>& dim1,
    std::vector<TensorShapeProto_Dimension>& dim2);

void assert_numpy_multibroadcastable(
    const std::vector<DimensionIR>& input1_sizes,
    const std::vector<DimensionIR>& input2_sizes);

void assertNotParams(const std::vector<DimensionIR>& sizes);

void assertInputsAvailable(const ArrayRef<Value*>& inputs, const char* name, uint64_t num_inputs);
} // namespace version_conversion
} // namespace ONNX_NAMESPACE
