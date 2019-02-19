#pragma once

#include "onnx/onnx-operators_pb.h"

namespace ONNX_NAMESPACE {

void SetAttrValue(float value, AttributeProto* out);
void SetAttrValue(int value, AttributeProto* out);
void SetAttrValue(const std::string& value, AttributeProto* out);
void SetAttrValue(const TensorProto& value, AttributeProto* out);

void SetAttrValue(std::vector<float> value, AttributeProto* out);
void SetAttrValue(std::vector<int> value, AttributeProto* out);
void SetAttrValue(const std::vector<std::string>& value, AttributeProto* out);
void SetAttrValue(const std::vector<TensorProto>& value, AttributeProto* out);

void SetAttrValue(const AttributeProto& value, AttributeProto* out);

} // namespace ONNX_NAMESPACE
