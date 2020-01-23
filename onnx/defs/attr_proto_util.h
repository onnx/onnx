// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include "onnx/onnx-operators_pb.h"

namespace ONNX_NAMESPACE {

AttributeProto MakeAttribute(const std::string& attr_name, const float& value);
AttributeProto MakeAttribute(const std::string& attr_name, const int64_t& value);
AttributeProto MakeAttribute(const std::string& attr_name, const std::string& value);
AttributeProto MakeAttribute(const std::string& attr_name, const TensorProto& value);
AttributeProto MakeAttribute(const std::string& attr_name, const GraphProto& value);
AttributeProto MakeAttribute(const std::string& attr_name, const std::vector<float>& values);
AttributeProto MakeAttribute(const std::string& attr_name, const std::vector<int64_t>& values);
AttributeProto MakeAttribute(const std::string& attr_name, const std::vector<std::string>& values);
AttributeProto MakeAttribute(const std::string& attr_name, const std::vector<TensorProto>& values);
AttributeProto MakeAttribute(const std::string& attr_name, const std::vector<GraphProto>& values);

AttributeProto MakeRefAttribute(const std::string& attr_name, AttributeProto_AttributeType type);

} // namespace ONNX_NAMESPACE
