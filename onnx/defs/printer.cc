/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/printer.h"

namespace ONNX_NAMESPACE {
namespace Utils {

std::ostream& operator<<(std::ostream& os, const TensorShapeProto_Dimension& dim) {
  if (dim.has_dim_value())
    os << dim.dim_value();
  else if (dim.has_dim_param())
    os << dim.dim_param();
  else
    os << "?";
  return os;
}

std::ostream& operator<<(std::ostream& os, const TensorShapeProto& shape) {
  const char* sep = "";
  const char* comma = ", ";
  os << "[";
  for (auto& dim : shape.dim()) {
    os << sep << dim;
    sep = comma;
  }
  os << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, const TypeProto_Tensor& tensortype) {
  os << PrimitiveTypeNameMap::ToString(tensortype.elem_type());
  if (tensortype.has_shape()) {
    if (tensortype.shape().dim_size() > 0)
      os << tensortype.shape();
  } else
    os << "[...]";

  return os;
}

std::ostream& operator<<(std::ostream& os, const TypeProto& type) {
  if (type.has_tensor_type())
    os << type.tensor_type();
  return os;
}

std::ostream& operator<<(std::ostream& os, const ValueInfoProto& value_info) {
  os << value_info.type() << " " << value_info.name();
  return os;
}

std::ostream& operator<<(std::ostream& os, const ValueInfoList& vilist) {
  const char* sep = "";
  const char* comma = ", ";
  os << "(";
  for (auto& vi : vilist) {
    os << sep << vi;
    sep = comma;
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const AttributeProto& attr) {
  const char* sep = "[";
  const char* comma = ", ";

  os << attr.name() << " = ";
  switch (attr.type()) {
    case AttributeProto_AttributeType_INT:
      os << attr.i();
      break;
    case AttributeProto_AttributeType_INTS:
      for (auto v : attr.ints()) {
        os << sep << v;
        sep = comma;
      }
      os << "]";
      break;
    case AttributeProto_AttributeType_FLOAT:
      os << attr.f();
      break;
    case AttributeProto_AttributeType_FLOATS:
      for (auto v : attr.floats()) {
        os << sep << v;
        sep = comma;
      }
      os << "]";
      break;
    case AttributeProto_AttributeType_STRING:
      os << "\"" << attr.s() << "\"";
      break;
    case AttributeProto_AttributeType_STRINGS:
      for (auto v : attr.strings()) {
        os << sep << "\"" << v << "\"";
        sep = comma;
      }
      os << "]";
      break;
    default:
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const AttrList& attrlist) {
  const char* sep = "";
  os << "<";
  for (auto& attr : attrlist) {
    os << sep << attr;
    sep = ", ";
  }
  os << ">";
  return os;
}

std::ostream& operator<<(std::ostream& os, const NodeProto& node) {
  const char* sep = "";
  for (auto& v : node.output()) {
    os << sep << v;
    sep = ", ";
  }
  os << " = " << node.op_type();
  if (node.attribute_size() > 0)
    os << node.attribute();
  os << "(";
  sep = "";
  for (auto& v : node.input()) {
    os << sep << v;
    sep = ", ";
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const NodeList& nodelist) {
  os << "{\n";
  for (auto& n : nodelist) {
    os << n << "\n";
  }
  os << "}\n";
  return os;
}

std::ostream& operator<<(std::ostream& os, const GraphProto& graph) {
  os << graph.name() << " " << graph.input() << " => " << graph.output() << " ";
  os << graph.node();
  return os;
}

} // namespace Utils
} // namespace ONNX_NAMESPACE