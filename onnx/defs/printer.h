/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <iostream>
#include <string>

#include "onnx/defs/parser.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {

std::ostream& operator<<(std::ostream& os, const TensorShapeProto_Dimension& proto);

std::ostream& operator<<(std::ostream& os, const TensorShapeProto& proto);

std::ostream& operator<<(std::ostream& os, const TypeProto_Tensor& proto);

std::ostream& operator<<(std::ostream& os, const TypeProto& proto);

std::ostream& operator<<(std::ostream& os, const TensorProto& proto);

std::ostream& operator<<(std::ostream& os, const ValueInfoProto& proto);

std::ostream& operator<<(std::ostream& os, const ValueInfoList& proto);

std::ostream& operator<<(std::ostream& os, const AttributeProto& proto);

std::ostream& operator<<(std::ostream& os, const AttrList& proto);

std::ostream& operator<<(std::ostream& os, const NodeProto& proto);

std::ostream& operator<<(std::ostream& os, const NodeList& proto);

std::ostream& operator<<(std::ostream& os, const GraphProto& proto);

std::ostream& operator<<(std::ostream& os, const FunctionProto& proto);

std::ostream& operator<<(std::ostream& os, const ModelProto& proto);

template <typename ProtoType>
std::string ProtoToString(const ProtoType& proto) {
  std::stringstream ss;
  ss << proto;
  return ss.str();
}

} // namespace ONNX_NAMESPACE
