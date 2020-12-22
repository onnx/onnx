#include <iostream>
#include "gtest/gtest.h"

#include "onnx/defs/parser.h"

using namespace ONNX_NAMESPACE::Utils;

namespace ONNX_NAMESPACE {
namespace Test {

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
  os << TensorProto_DataType_Name(static_cast<TensorProto_DataType>(tensortype.elem_type()));
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
  os << "{";
  for (auto& attr : attrlist) {
    os << sep << attr;
    sep = ", ";
  }
  os << "}";
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
  os << graph.name() << " (";
  os << ") ";
  os << graph.node();
  return os;
}

TEST(ParserTest, TypeTest) {
  TypeProto type;
  OnnxParser::Parse(type, "FLOAT[N]");
  std::cout << type << "\n";
  OnnxParser::Parse(type, "FLOAT");
  std::cout << type << "\n";
  OnnxParser::Parse(type, "FLOAT[]");
  std::cout << type << "\n";
  OnnxParser::Parse(type, "FLOAT[N,M,K]");
  std::cout << type << "\n";
  OnnxParser::Parse(type, "FLOAT[N,?,K]");
  std::cout << type << "\n";
}

TEST(ParserTest, NodeTest) {
  const char* code = "x = foo(y, z)";
  NodeProto n;
  OnnxParser::Parse(n, code);

  EXPECT_EQ(n.input_size(), 2);
  EXPECT_EQ(n.input(0), "y");
  EXPECT_EQ(n.input(1), "z");
  EXPECT_EQ(n.output_size(), 1);
  EXPECT_EQ(n.output(0), "x");
  EXPECT_EQ(n.op_type(), "foo");

  std::cout << n << "\n";
}

TEST(ParserTest, NodeListTest) {
  const char* code = R"ONNX(
{
    x = foo(y, z);
    w = bar(x, y);
}
)ONNX";

  GraphProto graph;
  OnnxParser::Parse(*graph.mutable_node(), code);

  EXPECT_EQ(graph.node_size(), 2);

  std::cout << graph.node();
}

TEST(ParserTest, AttrListTest) {
  const char* code = R"ONNX(
{
    x = 2,
    w = 3
}
)ONNX";

  NodeProto node;
  OnnxParser::Parse(*node.mutable_attribute(), code);

  std::cout << node.attribute();
}

TEST(ParserTest, NodeAttrTest1) {
  const char* code = "x = foo { a = 100, b = 200.5, c = \"astring\"} (y, z)";
  NodeProto n;
  OnnxParser::Parse(n, code);

  std::cout << n << "\n";
}

TEST(ParserTest, NodeAttrTest2) {
  const char* code = "x = foo { d = [5, 10], e = [0.55, 0.66], f = [\"str1\", \"str2\"] } (y, z)";
  NodeProto n;
  OnnxParser::Parse(n, code);

  std::cout << n << "\n";
}

// (float[N] y, float[N] z) => (float[N] w)

TEST(ParserTest, GraphTest) {
  const char* code = R"ONNX(
agraph ()
{
    x = foo(y, z);
    w = bar(x, y);
}
)ONNX";

  GraphProto graph;
  OnnxParser::Parse(graph, code);

  EXPECT_EQ(graph.node_size(), 2);

  std::cout << graph;
}

} // namespace Test
} // namespace ONNX_NAMESPACE