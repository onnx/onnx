// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <vector>

#include "onnx/onnx-operators_pb.h"

namespace ONNX_NAMESPACE {

AttributeProto MakeAttribute(std::string attr_name, float value);
AttributeProto MakeAttribute(std::string attr_name, int64_t value);
AttributeProto MakeAttribute(std::string attr_name, std::string value);
AttributeProto MakeAttribute(std::string attr_name, TensorProto value);
AttributeProto MakeAttribute(std::string attr_name, GraphProto value);
AttributeProto MakeAttribute(std::string attr_name, TypeProto value);
AttributeProto MakeAttribute(std::string attr_name, std::vector<float> values);
AttributeProto MakeAttribute(std::string attr_name, std::vector<int64_t> values);
AttributeProto MakeAttribute(std::string attr_name, std::vector<std::string> values);
AttributeProto MakeAttribute(std::string attr_name, std::vector<TensorProto> values);
AttributeProto MakeAttribute(std::string attr_name, std::vector<GraphProto> values);
AttributeProto MakeAttribute(std::string attr_name, std::vector<TypeProto> values);

// Make a "reference" attribute for a node in a function body.
// <attr_name> specifies the attribute name of both the function node and its
// function body node. They're using the same attribute name.
// <type> specifies the attribute type.
AttributeProto MakeRefAttribute(const std::string& attr_name, AttributeProto_AttributeType type);

// Make a "reference" attribute for a node in a function body.
// <attr_name> specifies the attribute name of the function body node.
// <referred_attr_name> specifies the referred attribute name of the function
// node.
// <type> specifies the attribute type.
AttributeProto MakeRefAttribute(
    const std::string& attr_name,
    const std::string& referred_attr_name,
    AttributeProto_AttributeType type);

} // namespace ONNX_NAMESPACE
