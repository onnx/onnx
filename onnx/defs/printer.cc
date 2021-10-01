/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/printer.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

template <typename Collection>
inline void print(std::ostream& os, const char* open, const char* separator, const char* close, Collection coll) {
  const char* sep = "";
  os << open;
  for (auto& elt : coll) {
    os << sep << elt;
    sep = separator;
  }
  os << close;
}

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
  print(os, "[", ",", "]", shape.dim());
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

std::ostream& operator<<(std::ostream& os, const TensorProto& tensor) {
  os << PrimitiveTypeNameMap::ToString(tensor.data_type());
  print(os, "[", ",", "]", tensor.dims());

  // TODO: does not yet handle raw_data or FLOAT16 or externally stored data.
  // TODO: does not yet handle name of tensor.
  switch (static_cast<TensorProto::DataType>(tensor.data_type())) {
    case TensorProto::DataType::TensorProto_DataType_INT8:
    case TensorProto::DataType::TensorProto_DataType_INT16:
    case TensorProto::DataType::TensorProto_DataType_INT32:
    case TensorProto::DataType::TensorProto_DataType_UINT8:
    case TensorProto::DataType::TensorProto_DataType_UINT16:
    case TensorProto::DataType::TensorProto_DataType_BOOL:
      print(os, " {", ",", "}", tensor.int32_data());
      break;
    case TensorProto::DataType::TensorProto_DataType_INT64:
      print(os, " {", ",", "}", tensor.int64_data());
      break;
    case TensorProto::DataType::TensorProto_DataType_UINT32:
    case TensorProto::DataType::TensorProto_DataType_UINT64:
      print(os, " {", ",", "}", tensor.uint64_data());
      break;
    case TensorProto::DataType::TensorProto_DataType_FLOAT:
      print(os, " {", ",", "}", tensor.float_data());
      break;
    case TensorProto::DataType::TensorProto_DataType_DOUBLE:
      print(os, " {", ",", "}", tensor.double_data());
      break;
    case TensorProto::DataType::TensorProto_DataType_STRING: {
      const char* sep = "{";
      for (auto& elt : tensor.string_data()) {
        os << sep << "\"" << elt << "\"";
        sep = ", ";
      }
      os << "}";
      break;
    }
    default:
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const ValueInfoProto& value_info) {
  os << value_info.type() << " " << value_info.name();
  return os;
}

std::ostream& operator<<(std::ostream& os, const ValueInfoList& vilist) {
  print(os, "(", ", ", ")", vilist);
  return os;
}

std::ostream& operator<<(std::ostream& os, const AttributeProto& attr) {
  os << attr.name() << " = ";
  switch (attr.type()) {
    case AttributeProto_AttributeType_INT:
      os << attr.i();
      break;
    case AttributeProto_AttributeType_INTS:
      print(os, "[", ", ", "]", attr.ints());
      break;
    case AttributeProto_AttributeType_FLOAT:
      os << attr.f();
      break;
    case AttributeProto_AttributeType_FLOATS:
      print(os, "[", ", ", "]", attr.floats());
      break;
    case AttributeProto_AttributeType_STRING:
      os << "\"" << attr.s() << "\"";
      break;
    case AttributeProto_AttributeType_STRINGS: {
      const char* sep = "[";
      for (auto& elt : attr.strings()) {
        os << sep << "\"" << elt << "\"";
        sep = ", ";
      }
      os << "]";
      break;
    }
    case AttributeProto_AttributeType_GRAPH:
      os << attr.g();
      break;
    default:
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const AttrList& attrlist) {
  print(os, "<", ", ", ">", attrlist);
  return os;
}

std::ostream& operator<<(std::ostream& os, const NodeProto& node) {
  print(os, "", ", ", "", node.output());
  os << " = " << node.op_type();
  if (node.attribute_size() > 0)
    os << node.attribute();
  print(os, "(", ", ", ")", node.input());
  return os;
}

std::ostream& operator<<(std::ostream& os, const NodeList& nodelist) {
  print(os, "{\n", "\n", "\n}\n", nodelist);
  return os;
}

std::ostream& operator<<(std::ostream& os, const GraphProto& graph) {
  os << graph.name() << " " << graph.input() << " => " << graph.output() << " ";
  os << graph.node();
  return os;
}

std::ostream& operator<<(std::ostream& os, const OperatorSetIdProto& opset) {
  os << "\"" << opset.domain() << "\" : " << opset.version();
  return os;
}

std::ostream& operator<<(std::ostream& os, const FunctionProto& fn) {
  os << "<\n";
  os << "  " << "domain: \"" << fn.domain() << "\",\n";
  os << "  " << "opset_import: ";
  print (os, "[", ",", "]", fn.opset_import());
  os << "\n>\n";
  os << fn.name() << " ";
  if (fn.attribute_size() > 0)
    print(os, "<", ",", ">", fn.attribute());
  print(os, "(", ", ", ")", fn.input());
  os << " => ";
  print(os, "(", ", ", ")", fn.output());
  os << "\n";
  os << fn.node();
  return os;
}

} // namespace ONNX_NAMESPACE