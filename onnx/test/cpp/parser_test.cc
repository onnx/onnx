#include <iostream>
#include "gtest/gtest.h"

#include "onnx/defs/parser.h"

using namespace ONNX_NAMESPACE::Utils;

namespace ONNX_NAMESPACE {
namespace Test {

std::ostream& operator<<(std::ostream& os, const AttributeProto& attr) {
  os << attr.name() << " = ";
  switch (attr.type()) {
    case AttributeProto_AttributeType_INT:
      os << attr.i();
      break;
    case AttributeProto_AttributeType_FLOAT:
      os << attr.f();
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

TEST(ParserTest, NodeAttrTest) {
  const char* code = "x = foo { a = 100, b = 200.5 } (y, z)";
  NodeProto n;
  OnnxParser::Parse(n, code);

  std::cout << n << "\n";
}

} // namespace Test
} // namespace ONNX_NAMESPACE