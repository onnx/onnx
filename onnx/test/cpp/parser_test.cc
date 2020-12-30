#include "gtest/gtest.h"

#include "onnx/defs/parser.h"
#include "onnx/defs/printer.h"

using namespace ONNX_NAMESPACE::Utils;

namespace ONNX_NAMESPACE {
namespace Test {

TEST(ParserTest, TypeTest) {
  TypeProto type;

  // 1-dimensional tensor type with symbolic dimension:
  OnnxParser::Parse(type, "FLOAT[N]");
  EXPECT_TRUE(type.has_tensor_type());
  int float_type = static_cast<int>(TensorProto_DataType::TensorProto_DataType_FLOAT);
  EXPECT_EQ(type.tensor_type().elem_type(), float_type);
  EXPECT_TRUE(type.tensor_type().has_shape());
  EXPECT_EQ(type.tensor_type().shape().dim_size(), 1);
  EXPECT_EQ(type.tensor_type().shape().dim(0).dim_param(), "N");

  // scalar type:
  OnnxParser::Parse(type, "FLOAT");
  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_EQ(type.tensor_type().elem_type(), float_type);
  EXPECT_TRUE(type.tensor_type().has_shape());
  EXPECT_EQ(type.tensor_type().shape().dim_size(), 0);

  // tensor type with unknown rank:
  OnnxParser::Parse(type, "FLOAT[]");
  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_EQ(type.tensor_type().elem_type(), float_type);
  EXPECT_FALSE(type.tensor_type().has_shape());

  // 3-dimensional tensor
  OnnxParser::Parse(type, "FLOAT[N,M,K]");
  EXPECT_EQ(type.tensor_type().shape().dim_size(), 3);

  // Unspecified dimension (neither symbolic nor constant)
  OnnxParser::Parse(type, "FLOAT[N,?,K]");
  EXPECT_FALSE(type.tensor_type().shape().dim(1).has_dim_param());
  EXPECT_FALSE(type.tensor_type().shape().dim(1).has_dim_value());
}

TEST(ParserTest, AttributeTest) {
  AttributeProto attr;

  OnnxParser::Parse(attr, "x = 2");
  EXPECT_EQ(attr.name(), "x");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  EXPECT_EQ(attr.i(), 2);

  OnnxParser::Parse(attr, "x = 0.625");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT);
  EXPECT_FLOAT_EQ(attr.f(), 0.625);

  OnnxParser::Parse(attr, "x = [2, 4, 6]");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  EXPECT_EQ(attr.ints_size(), 3);

  OnnxParser::Parse(attr, "x = [0.125, 0.625]");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS);
  EXPECT_EQ(attr.floats_size(), 2);

  OnnxParser::Parse(attr, "x = \"astring\"");
  EXPECT_EQ(attr.name(), "x");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
  EXPECT_EQ(attr.s(), "astring");

  OnnxParser::Parse(attr, "x = [\"abc\", \"def\"]");
  EXPECT_EQ(attr.type(), AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS);
}

TEST(ParserTest, AttrListTest) {
  const char* code = R"ONNX(
<
    x = 2,
    w = 3
>
)ONNX";

  AttrList attributes;
  OnnxParser::Parse(attributes, code);
  EXPECT_EQ(attributes.size(), 2);
  EXPECT_EQ(attributes.Get(0).name(), "x");
  EXPECT_EQ(attributes.Get(1).name(), "w");
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
  EXPECT_EQ(graph.node(0).op_type(), "foo");
  EXPECT_EQ(graph.node(1).op_type(), "bar");
}

TEST(ParserTest, NodeAttrTest1) {
  const char* code = "x = foo <a = 100, b = 200.5, c = \"astring\"> (y, z)";
  NodeProto n;
  OnnxParser::Parse(n, code);

  EXPECT_EQ(n.attribute_size(), 3);
  EXPECT_EQ(n.attribute(0).name(), "a");
  EXPECT_EQ(n.attribute(1).name(), "b");
  EXPECT_EQ(n.attribute(2).name(), "c");
}

TEST(ParserTest, NodeAttrTest2) {
  const char* code = "x = foo <d = [5, 10], e = [0.55, 0.66], f = [\"str1\", \"str2\"]> (y, z)";
  NodeProto n;
  OnnxParser::Parse(n, code);
  EXPECT_EQ(n.attribute_size(), 3);
}

TEST(ParserTest, GraphTest) {
  const char* code = R"ONNX(
agraph (FLOAT[N] y, FLOAT[N] z) => (FLOAT[N] w)
{
    x = foo(y, z);
    w = bar(x, y);
}
)ONNX";

  GraphProto graph;
  OnnxParser::Parse(graph, code);

  EXPECT_EQ(graph.input_size(), 2);
  EXPECT_EQ(graph.output_size(), 1);
  EXPECT_EQ(graph.node_size(), 2);
}

} // namespace Test
} // namespace ONNX_NAMESPACE