#pragma once

#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"

namespace ONNX_NAMESPACE {

void AssertAttributeProtoTypeAndLength(
    const AttributeProto* attr_proto,
    int expected_length,
    TensorProto_DataType expected_type,
    bool required);

} // namespace ONNX_NAMESPACE
