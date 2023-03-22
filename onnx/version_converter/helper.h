/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Helper Methods for Adapters

#pragma once

#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {
bool HasAttribute(NodeProto* node_proto, const std::string& name);

template <typename T>
std::vector<T> protobuf_repeated_fields_to_vector(const google::protobuf::RepeatedPtrField<T>& repeated);

int check_numpy_unibroadcastable_and_require_broadcast(
    std::vector<TensorShapeProto_Dimension>& dim1,
    std::vector<TensorShapeProto_Dimension>& dim2);

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
