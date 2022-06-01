/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/printer.h"
#include <iomanip>
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

class ProtoPrinter {
 public:
  ProtoPrinter(std::ostream& os) : output(os) {}

  void print(const TensorShapeProto_Dimension& dim);

  void print(const TensorShapeProto& shape);

  void print(const TypeProto_Tensor& tensortype);

  void print(const TypeProto& type);

  void print(const TypeProto_Sequence& seqType);

  void print(const TypeProto_Map& mapType);

  void print(const TypeProto_Optional& optType);

  void print(const TypeProto_SparseTensor sparseType);

  void print(const TensorProto& tensor);

  void print(const ValueInfoProto& value_info);

  void print(const ValueInfoList& vilist);

  void print(const AttributeProto& attr);

  void print(const AttrList& attrlist);

  void print(const NodeProto& node);

  void print(const NodeList& nodelist);

  void print(const GraphProto& graph);

  void print(const FunctionProto& fn);

  void print(const OperatorSetIdProto& opset);

 private:
  template <typename T>
  inline void print(T prim) {
    output << prim;
  }

  template <typename Collection>
  inline void print(const char* open, const char* separator, const char* close, Collection coll) {
    const char* sep = "";
    output << open;
    for (auto& elt : coll) {
      output << sep;
      print(elt);
      sep = separator;
    }
    output << close;
  }

  std::ostream& output;
  int indent_level = 3;

  void indent() {
    indent_level += 3;
  }

  void outdent() {
    indent_level -= 3;
  }
};

void ProtoPrinter::print(const TensorShapeProto_Dimension& dim) {
  if (dim.has_dim_value())
    output << dim.dim_value();
  else if (dim.has_dim_param())
    output << dim.dim_param();
  else
    output << "?";
}

void ProtoPrinter::print(const TensorShapeProto& shape) {
  print("[", ",", "]", shape.dim());
}

void ProtoPrinter::print(const TypeProto_Tensor& tensortype) {
  output << PrimitiveTypeNameMap::ToString(tensortype.elem_type());
  if (tensortype.has_shape()) {
    if (tensortype.shape().dim_size() > 0)
      print(tensortype.shape());
  } else
    output << "[]";
}

void ProtoPrinter::print(const TypeProto_Sequence& seqType) {
  output << "seq(";
  print(seqType.elem_type());
  output << ")";
}

void ProtoPrinter::print(const TypeProto_Map& mapType) {
  output << "map(" << PrimitiveTypeNameMap::ToString(mapType.key_type()) << ", ";
  print(mapType.value_type());
  output << ")";
}

void ProtoPrinter::print(const TypeProto_Optional& optType) {
  output << "optional(";
  print(optType.elem_type());
  output << ")";
}

void ProtoPrinter::print(const TypeProto_SparseTensor sparseType) {
  output << "sparse_tensor(" << PrimitiveTypeNameMap::ToString(sparseType.elem_type());
  if (sparseType.has_shape()) {
    if (sparseType.shape().dim_size() > 0)
      print(sparseType.shape());
  } else
    output << "[]";

  output << ")";
}

void ProtoPrinter::print(const TypeProto& type) {
  if (type.has_tensor_type())
    print(type.tensor_type());
  else if (type.has_sequence_type())
    print(type.sequence_type());
  else if (type.has_map_type())
    print(type.map_type());
  else if (type.has_optional_type())
    print(type.optional_type());
  else if (type.has_sparse_tensor_type())
    print(type.sparse_tensor_type());
}

void ProtoPrinter::print(const TensorProto& tensor) {
  output << PrimitiveTypeNameMap::ToString(tensor.data_type());
  if (tensor.dims_size() > 0)
    print("[", ",", "]", tensor.dims());

  if (!tensor.name().empty()) {
    output << " " << tensor.name();
  }
  // TODO: does not yet handle all types or externally stored data.
  if (tensor.has_raw_data()) {
    switch (static_cast<TensorProto::DataType>(tensor.data_type())) {
      case TensorProto::DataType::TensorProto_DataType_INT32:
        print(" {", ",", "}", ParseData<int32_t>(&tensor));
        break;
      case TensorProto::DataType::TensorProto_DataType_INT64:
        print(" {", ",", "}", ParseData<int64_t>(&tensor));
        break;
      case TensorProto::DataType::TensorProto_DataType_FLOAT:
        print(" {", ",", "}", ParseData<float>(&tensor));
        break;
      case TensorProto::DataType::TensorProto_DataType_DOUBLE:
        print(" {", ",", "}", ParseData<double>(&tensor));
        break;
      default:
        output << "..."; // ParseData not instantiated for other types.
        break;
    }
  } else {
    switch (static_cast<TensorProto::DataType>(tensor.data_type())) {
      case TensorProto::DataType::TensorProto_DataType_INT8:
      case TensorProto::DataType::TensorProto_DataType_INT16:
      case TensorProto::DataType::TensorProto_DataType_INT32:
      case TensorProto::DataType::TensorProto_DataType_UINT8:
      case TensorProto::DataType::TensorProto_DataType_UINT16:
      case TensorProto::DataType::TensorProto_DataType_BOOL:
        print(" {", ",", "}", tensor.int32_data());
        break;
      case TensorProto::DataType::TensorProto_DataType_INT64:
        print(" {", ",", "}", tensor.int64_data());
        break;
      case TensorProto::DataType::TensorProto_DataType_UINT32:
      case TensorProto::DataType::TensorProto_DataType_UINT64:
        print(" {", ",", "}", tensor.uint64_data());
        break;
      case TensorProto::DataType::TensorProto_DataType_FLOAT:
        print(" {", ",", "}", tensor.float_data());
        break;
      case TensorProto::DataType::TensorProto_DataType_DOUBLE:
        print(" {", ",", "}", tensor.double_data());
        break;
      case TensorProto::DataType::TensorProto_DataType_STRING: {
        const char* sep = "{";
        for (auto& elt : tensor.string_data()) {
          output << sep << "\"" << elt << "\"";
          sep = ", ";
        }
        output << "}";
        break;
      }
      default:
        break;
    }
  }
}

void ProtoPrinter::print(const ValueInfoProto& value_info) {
  print(value_info.type());
  output << " " << value_info.name();
}

void ProtoPrinter::print(const ValueInfoList& vilist) {
  print("(", ", ", ")", vilist);
}

void ProtoPrinter::print(const AttributeProto& attr) {
  // Special case of attr-ref:
  if (attr.has_ref_attr_name()) {
    output << attr.name() << " : " << AttributeTypeNameMap::ToString(attr.type()) << " = @" << attr.ref_attr_name();
  }
  // General case:
  output << attr.name() << " = ";
  switch (attr.type()) {
    case AttributeProto_AttributeType_INT:
      output << attr.i();
      break;
    case AttributeProto_AttributeType_INTS:
      print("[", ", ", "]", attr.ints());
      break;
    case AttributeProto_AttributeType_FLOAT:
      output << attr.f();
      break;
    case AttributeProto_AttributeType_FLOATS:
      print("[", ", ", "]", attr.floats());
      break;
    case AttributeProto_AttributeType_STRING:
      output << "\"" << attr.s() << "\"";
      break;
    case AttributeProto_AttributeType_STRINGS: {
      const char* sep = "[";
      for (auto& elt : attr.strings()) {
        output << sep << "\"" << elt << "\"";
        sep = ", ";
      }
      output << "]";
      break;
    }
    case AttributeProto_AttributeType_GRAPH:
      indent();
      print(attr.g());
      outdent();
      break;
    case AttributeProto_AttributeType_GRAPHS:
      indent();
      print("[", ", ", "]", attr.graphs());
      outdent();
      break;
    case AttributeProto_AttributeType_TENSOR:
      print(attr.t());
      break;
    case AttributeProto_AttributeType_TENSORS:
      print("[", ", ", "]", attr.tensors());
      break;
    default:
      break;
  }
}

void ProtoPrinter::print(const AttrList& attrlist) {
  print(" <", ", ", ">", attrlist);
}

void ProtoPrinter::print(const NodeProto& node) {
  output << std::setw(indent_level) << ' ';
  print("", ", ", "", node.output());
  output << " = " << node.op_type();
  bool has_subgraph = false;
  for (auto attr : node.attribute())
    if (attr.has_g() || (attr.graphs_size() > 0))
      has_subgraph = true;
  if ((!has_subgraph) && (node.attribute_size() > 0))
    print(node.attribute());
  print(" (", ", ", ")", node.input());
  if ((has_subgraph) && (node.attribute_size() > 0))
    print(node.attribute());
  output << "\n";
}

void ProtoPrinter::print(const NodeList& nodelist) {
  const char* sep = "";
  output << "{\n";
  for (auto& node : nodelist) {
    print(node);
  }
  if (indent_level > 3)
    output << std::setw(indent_level - 3) << "   ";
  output << "}";
}

void ProtoPrinter::print(const GraphProto& graph) {
  output << graph.name() << " " << graph.input() << " => " << graph.output() << " ";
  print(graph.node());
}

void ProtoPrinter::print(const OperatorSetIdProto& opset) {
  output << "\"" << opset.domain() << "\" : " << opset.version();
}

void ProtoPrinter::print(const FunctionProto& fn) {
  output << "<\n";
  output << "  "
         << "domain: \"" << fn.domain() << "\",\n";
  output << "  "
         << "opset_import: ";
  print("[", ",", "]", fn.opset_import());
  output << "\n>\n";
  output << fn.name() << " ";
  if (fn.attribute_size() > 0)
    print("<", ",", ">", fn.attribute());
  print("(", ", ", ")", fn.input());
  output << " => ";
  print("(", ", ", ")", fn.output());
  output << "\n";
  print(fn.node());
}

#define DEF_OP(T) \
  std::ostream& operator<<(std::ostream& os, const T& proto) { \
    ProtoPrinter printer(os); \
    printer.print(proto); \
    return os; \
  };

DEF_OP(TensorShapeProto_Dimension)

DEF_OP(TensorShapeProto)

DEF_OP(TypeProto_Tensor)

DEF_OP(TypeProto)

DEF_OP(TensorProto)

DEF_OP(ValueInfoProto)

DEF_OP(ValueInfoList)

DEF_OP(AttributeProto)

DEF_OP(AttrList)

DEF_OP(NodeProto)

DEF_OP(NodeList)

DEF_OP(GraphProto)

DEF_OP(FunctionProto)

} // namespace ONNX_NAMESPACE