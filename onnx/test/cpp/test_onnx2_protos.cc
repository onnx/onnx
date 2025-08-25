// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>  // NOLINT(build/c++17)
#include <string>
#include <vector>

#include "onnx/onnx2/cpu/onnx2.h"
#include "onnx/onnx2/cpu/onnx2_helper.h"
#include "onnx/onnx2/cpu/common_helpers.h"

using namespace onnx2;

TEST(onnx2_string, RefString_Constructors) {
  utils::RefString original("test", 4);
  utils::RefString copied(original);
  EXPECT_EQ(copied.size(), 4);
  EXPECT_EQ(copied.data(), original.data());
  EXPECT_EQ(copied, original);

  const char *text = "hello";
  utils::RefString rs(text, 5);
  EXPECT_EQ(rs.size(), 5);
  EXPECT_EQ(rs.data(), text);
}

TEST(onnx2_string, RefString_Assignment) {
  utils::RefString a("abc", 3);
  utils::RefString b("xyz", 3);
  b = a;
  EXPECT_EQ(b.data(), a.data());
  EXPECT_EQ(b.size(), 3);

  utils::String s("def", 3);
  utils::RefString c("123", 3);
  c = s;
  EXPECT_EQ(c.data(), s.data());
  EXPECT_EQ(c.size(), 3);
}

TEST(onnx2_string, RefString_Methods) {
  utils::RefString a("hello", 5);
  EXPECT_EQ(a.size(), 5);
  EXPECT_EQ(a.c_str(), a.data());
  EXPECT_FALSE(a.empty());
  utils::RefString empty(nullptr, 0);
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(a[0], 'h');
  EXPECT_EQ(a[4], 'o');
}

TEST(onnx2_string, RefString_Equality) {
  utils::RefString a("test", 4);
  utils::RefString b("test", 4);
  utils::RefString c("diff", 4);
  utils::String d("test", 4);
  std::string e("test");
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_TRUE(a == d);
  EXPECT_TRUE(a == e);
  EXPECT_TRUE(a == "test");
  EXPECT_FALSE(a == "different");
  utils::RefString empty(nullptr, 0);
  EXPECT_TRUE(empty == "");
  EXPECT_TRUE(empty == nullptr);
}

TEST(onnx2_string, RefString_Inequality) {
  utils::RefString a("test", 4);
  utils::RefString b("test", 4);
  utils::RefString c("diff", 4);
  utils::String d("test", 4);
  utils::String e("diff", 4);
  std::string f("test");
  std::string g("diff");
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);
  EXPECT_FALSE(a != d);
  EXPECT_TRUE(a != e);
  EXPECT_FALSE(a != f);
  EXPECT_TRUE(a != g);
  EXPECT_FALSE(a != "test");
  EXPECT_TRUE(a != "diff");
}

TEST(onnx2_string, RefString_AsString) {
  utils::RefString a("hello", 5);
  std::string str = a.as_string();
  EXPECT_EQ(str, "hello");

  utils::RefString empty(nullptr, 0);
  std::string emptyStr = empty.as_string();
  EXPECT_EQ(emptyStr, "");
}

TEST(onnx2_string, String_Constructors) {
  utils::String defaultStr;
  EXPECT_EQ(defaultStr.size(), 0);
  EXPECT_EQ(defaultStr.data(), nullptr);
  EXPECT_TRUE(defaultStr.empty());
  utils::RefString ref("test", 4);
  utils::String fromRef(ref);
  EXPECT_EQ(fromRef.size(), 4);
  EXPECT_NE(fromRef.data(), ref.data());
  EXPECT_EQ(fromRef, ref);

  utils::String fromCharPtr("hello", 5);
  EXPECT_EQ(fromCharPtr.size(), 5);
  EXPECT_EQ(fromCharPtr, "hello");

  utils::String withNull("abc\0", 4);
  EXPECT_EQ(withNull.size(), 3);
  EXPECT_EQ(withNull, "abc");

  std::string stdStr = "world";
  utils::String fromStdStr(stdStr);
  EXPECT_EQ(fromStdStr.size(), 5);
  EXPECT_EQ(fromStdStr, stdStr);
}

TEST(onnx2_string, String_Assignment) {
  utils::String s;

  s = "abc";
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s, "abc");

  utils::RefString ref("def", 3);
  s = ref;
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s, ref);

  utils::String other("xyz", 3);
  s = other;
  EXPECT_EQ(s.size(), 3);
  EXPECT_EQ(s, other);
  EXPECT_NE(s.data(), other.data());

  std::string stdStr = "hello";
  s = stdStr;
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s, stdStr);
}

TEST(onnx2_string, String_Methods) {
  utils::String s("hello", 5);
  EXPECT_EQ(s.size(), 5);
  EXPECT_NE(s.data(), nullptr);
  EXPECT_FALSE(s.empty());
  utils::String empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(s[0], 'h');
  EXPECT_EQ(s[4], 'o');
  s.clear();
  EXPECT_EQ(s.size(), 0);
  EXPECT_EQ(s.data(), nullptr);
  EXPECT_TRUE(s.empty());
}

TEST(onnx2_string, String_Equality) {
  utils::String a("test", 4);
  utils::String b("test", 4);
  utils::String c("diff", 4);
  utils::RefString d("test", 4);
  std::string e("test");

  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a == c);
  EXPECT_TRUE(a == d);
  EXPECT_TRUE(a == e);
  EXPECT_TRUE(a == "test");
  EXPECT_FALSE(a == "different");
  utils::String empty;
  EXPECT_TRUE(empty == "");
  EXPECT_TRUE(empty == nullptr);
}

TEST(onnx2_string, String_Inequality) {
  utils::String a("test", 4);
  utils::String b("test", 4);
  utils::String c("diff", 4);
  utils::RefString d("test", 4);
  utils::RefString e("diff", 4);
  std::string f("test");
  std::string g("diff");

  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);

  EXPECT_FALSE(a != d);
  EXPECT_TRUE(a != e);

  EXPECT_FALSE(a != f);
  EXPECT_TRUE(a != g);

  EXPECT_FALSE(a != "test");
  EXPECT_TRUE(a != "diff");
}

TEST(onnx2_string, String_AsString) {
  utils::String a("hello", 5);
  std::string str = a.as_string();
  EXPECT_EQ(str, "hello");

  utils::String empty;
  std::string emptyStr = empty.as_string();
  EXPECT_EQ(emptyStr, "");
}

TEST(onnx2_string, String_EdgeCases) {
  utils::String empty("");
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.size(), 0);

  utils::String null(nullptr, 0);
  EXPECT_TRUE(null.empty());
  EXPECT_EQ(null.size(), 0);

  utils::String withNulls("abc\0def", 7);
  EXPECT_EQ(withNulls.size(), 7);
  EXPECT_EQ(withNulls[3], '\0');
  EXPECT_EQ(withNulls[4], 'd');
}

TEST(onnx2_string, RefString) {
  utils::RefString a("iii", 3);
  EXPECT_EQ(a.size(), 3);
  EXPECT_FALSE(a.empty());
  EXPECT_EQ(a, a);
  EXPECT_EQ(a, "iii");
}

TEST(onnx2_string, RefString_ConstructorFromConstCharPtr) {
  const char *text = "hello world";
  utils::RefString rs(text, 11);
  EXPECT_EQ(rs.size(), 11);
  EXPECT_EQ(rs.data(), text);
  EXPECT_FALSE(rs.empty());
  EXPECT_EQ(rs, "hello world");

  utils::RefString empty_rs("", 0);
  EXPECT_EQ(empty_rs.size(), 0);
  EXPECT_TRUE(empty_rs.empty());
  EXPECT_EQ(empty_rs, "");

  utils::RefString null_rs(nullptr, 0);
  EXPECT_EQ(null_rs.size(), 0);
  EXPECT_TRUE(null_rs.empty());
  EXPECT_EQ(null_rs.data(), nullptr);
}

TEST(onnx2_string, RefString_AssignmentVariations) {
  utils::RefString source("source", 6);
  utils::RefString target("target", 6);
  EXPECT_EQ(target, "target");
  target = source;
  EXPECT_EQ(target, "source");
  EXPECT_EQ(target.data(), source.data());
  EXPECT_EQ(target.size(), source.size());

  utils::RefString empty(nullptr, 0);
  utils::RefString non_empty("data", 4);
  non_empty = empty;
  EXPECT_TRUE(non_empty.empty());
  EXPECT_EQ(non_empty.size(), 0);
  EXPECT_EQ(non_empty.data(), nullptr);
}

TEST(onnx2_string, String_ConstructorFromConstCharPtrVariations) {
  utils::String s1("test string", 11);
  EXPECT_EQ(s1.size(), 11);
  EXPECT_EQ(s1, "test string");

  utils::String s2("embedded\0null", 13);
  EXPECT_EQ(s2.size(), 13);
  EXPECT_EQ(s2[8], '\0');
  EXPECT_EQ(s2[9], 'n');

  utils::String s3("", 0);
  EXPECT_EQ(s3.size(), 0);
  EXPECT_TRUE(s3.empty());

  utils::String s4(nullptr, 0);
  EXPECT_EQ(s4.size(), 0);
  EXPECT_TRUE(s4.empty());
}

TEST(onnx2_string, String_ConstructorFromStdString) {
  std::string std_str = "hello std::string";
  utils::String s1(std_str);
  EXPECT_EQ(s1.size(), std_str.size());
  EXPECT_EQ(s1, std_str);

  std::string empty_str = "";
  utils::String s2(empty_str);
  EXPECT_EQ(s2.size(), 0);
  EXPECT_TRUE(s2.empty());

  std::string null_str(10, '\0');
  utils::String s3(null_str);
  // String loses the last character if it is a null terminator.
  EXPECT_EQ(s3.size(), 9);
  for (size_t i = 0; i < 9; i++) {
    EXPECT_EQ(s3[i], '\0');
  }
}

TEST(onnx2_string, String_ConstructorFromRefString) {
  utils::RefString ref1("reference data", 14);
  utils::String s1(ref1);
  EXPECT_EQ(s1.size(), ref1.size());
  EXPECT_EQ(s1, ref1);
  EXPECT_NE(s1.data(), ref1.data());

  utils::RefString ref2(nullptr, 0);
  utils::String s2(ref2);
  EXPECT_EQ(s2.size(), 0);
  EXPECT_TRUE(s2.empty());

  char data[5] = {'a', '\0', 'b', '\0', 'c'};
  utils::RefString ref3(data, 5);
  utils::String s3(ref3);
  EXPECT_EQ(s3.size(), 5);
  EXPECT_EQ(s3[0], 'a');
  EXPECT_EQ(s3[1], '\0');
  EXPECT_EQ(s3[2], 'b');
  EXPECT_EQ(s3[3], '\0');
  EXPECT_EQ(s3[4], 'c');
}

TEST(onnx2_string, String_CopyConstructor) {
  utils::String original("original data", 13);
  utils::String copy(original);
  EXPECT_EQ(copy.size(), original.size());
  EXPECT_EQ(copy, original);
  EXPECT_NE(copy.data(), original.data());

  original = "changed data";
  EXPECT_EQ(copy, "original data");
  EXPECT_NE(copy, original);

  utils::String empty_original;
  utils::String empty_copy(empty_original);
  EXPECT_EQ(empty_copy.size(), 0);
  EXPECT_TRUE(empty_copy.empty());
}

TEST(onnx2_string, String_MoveConstructor) {
  utils::String original("move this data", 14);
  const char *original_data = original.data();
  utils::String moved(std::move(original));

  EXPECT_EQ(moved.size(), 14);
  EXPECT_EQ(moved, "move this data");
  EXPECT_EQ(moved.data(), original_data);

  EXPECT_TRUE(original.empty() || original.size() == 0);
}

TEST(onnx2_string, String_AssignmentOperators) {
  utils::String s;

  s = "1234567890123456789";
  EXPECT_EQ(s.size(), 19);
  EXPECT_EQ(s, "1234567890123456789");

  utils::RefString ref("1234567890123456789012", 22);
  s = ref;
  EXPECT_EQ(s.size(), 22);
  EXPECT_EQ(s, ref);
  EXPECT_NE(s.data(), ref.data());

  utils::String other("A234567890123456789", 19);
  s = other;
  EXPECT_EQ(s.size(), 19);
  EXPECT_EQ(s, other);
  EXPECT_NE(s.data(), other.data());

  std::string std_str = "assigned from std::string";
  s = std_str;
  EXPECT_EQ(s.size(), std_str.size());
  EXPECT_EQ(s, std_str);

  s = s;
  EXPECT_EQ(s, "assigned from std::string");
}

TEST(onnx2_string, String_SelfAssignmentSafety) {
  utils::String s("12345678901234567890", 19);

  s = s;
  EXPECT_EQ(s.size(), 19);
  EXPECT_EQ(s, "1234567890123456789");
  EXPECT_EQ(s, "1234567890123456789");

  utils::String *ptr = &s;
  *ptr = *ptr;
  EXPECT_EQ(*ptr, "1234567890123456789");
}

TEST(onnx2_string, RefString_EqualityEdgeCases) {
  char data1[] = {'t', 'e', 's', 't', '\0', '!'};
  char data2[] = {'t', 'e', 's', 't', '\0', '?'};

  utils::RefString rs1(data1, 6);
  utils::RefString rs2(data2, 6);
  utils::RefString rs3(data1, 4);

  EXPECT_NE(rs1, rs2);
  EXPECT_NE(rs1, rs3);

  utils::RefString null_rs(nullptr, 0);
  utils::RefString empty_rs("", 0);

  EXPECT_EQ(null_rs, empty_rs);
  EXPECT_EQ(null_rs, nullptr);
  EXPECT_EQ(null_rs, "");
  EXPECT_NE(rs1, nullptr);
  EXPECT_NE(rs1, "");
}

TEST(onnx2_string, String_EqualityEdgeCases) {
  utils::String s1("test\0!", 6);
  utils::String s2("test\0?", 6);
  utils::String s3("test", 4);

  EXPECT_NE(s1, s2);
  EXPECT_NE(s1, s3);

  utils::String empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty, "");
  EXPECT_EQ(empty, nullptr);
  EXPECT_NE(s1, nullptr);
  EXPECT_NE(s1, "");

  utils::RefString rs("test", 4);
  EXPECT_EQ(s3, rs);
  EXPECT_NE(s1, rs);
}

TEST(onnx2_string, String_NullVersusSizeZero) {
  utils::String null_string;
  EXPECT_TRUE(null_string.empty());
  EXPECT_EQ(null_string.size(), 0);
  EXPECT_EQ(null_string.data(), nullptr);

  utils::String empty_string("", 0);
  EXPECT_TRUE(empty_string.empty());
  EXPECT_EQ(empty_string.size(), 0);

  EXPECT_EQ(null_string, empty_string);
  EXPECT_EQ(null_string, "");
  EXPECT_EQ(empty_string, "");
}

TEST(onnx2_string, RefString_AsStringEdgeCases) {
  utils::RefString rs1("regular string", 13);
  std::string s1 = rs1.as_string();
  EXPECT_EQ(s1, "regular strin");

  char data[] = {'t', 'e', 's', 't', '\0', '!'};
  utils::RefString rs2(data, 6);
  std::string s2 = rs2.as_string();
  EXPECT_EQ(s2.size(), 6);
  EXPECT_EQ(s2[4], '\0');

  utils::RefString null_rs(nullptr, 0);
  std::string s3 = null_rs.as_string();
  EXPECT_TRUE(s3.empty());
  EXPECT_EQ(s3, "");
}

TEST(onnx2_string, String_AsStringEdgeCases) {
  utils::String s1("1234567890123", 13);
  std::string std_s1 = s1.as_string();
  EXPECT_EQ(std_s1, "1234567890123");

  utils::String s2("test\0!", 6);
  std::string std_s2 = s2.as_string();
  EXPECT_EQ(std_s2.size(), 6);
  EXPECT_EQ(std_s2[4], '\0');

  // String vide
  utils::String empty;
  std::string std_s3 = empty.as_string();
  EXPECT_TRUE(std_s3.empty());
  EXPECT_EQ(std_s3, "");
}

TEST(onnx2_string, String) {
  utils::String a("iii", 3);
  EXPECT_EQ(a.size(), 3);
  EXPECT_FALSE(a.empty());
  EXPECT_EQ(a, a);
  EXPECT_EQ(a, "iii");
  std::string s("iii");
  utils::String b(s);
  EXPECT_EQ(b.size(), 3);
  EXPECT_FALSE(b.empty());
  EXPECT_EQ(b, a);
  EXPECT_EQ(b, "iii");
}

TEST(onnx2_proto, TensorProtoName1) {
  TensorProto tp;
  EXPECT_EQ(tp.name_.data(), nullptr);
  EXPECT_EQ(tp.name_.size(), 0);
  EXPECT_EQ(tp.ref_name().data(), nullptr);
  EXPECT_EQ(tp.ref_name().size(), 0);
  std::string name("test");
  tp.name_ = name;
  EXPECT_EQ(tp.name_.size(), 4);
  EXPECT_NE(tp.name_.data(), nullptr);
  EXPECT_EQ(tp.name_.data()[0], 't');
  EXPECT_EQ(tp.order_name(), 8);
}

TEST(onnx2_proto, TensorProtoName2) {
  TensorProto tp;
  EXPECT_EQ(tp.name_.data(), nullptr);
  EXPECT_EQ(tp.name_.size(), 0);
  EXPECT_EQ(tp.ref_name().data(), nullptr);
  EXPECT_EQ(tp.ref_name().size(), 0);
  std::string name("test");
  tp.set_name(name);
  EXPECT_EQ(tp.name_.size(), 4);
  EXPECT_NE(tp.name_.data(), nullptr);
  EXPECT_EQ(tp.name_.data()[0], 't');
  std::string check = tp.name_.as_string();
  EXPECT_EQ(name, check);
  std::string check4 = tp.ref_name().as_string();
  EXPECT_EQ(name, check4);
  name = "TEST2";
  tp.set_name(name);
  std::string check2 = tp.name_.as_string();
  EXPECT_EQ(name, check2);
}

TEST(onnx2_proto, TensorProtoNameStringToString1) {
  {
    TensorProto tp;
    tp.name_ = "test";
    if (tp.ref_name().size() == 4) {
      TensorProto tp2;
      tp2.set_name(tp.ref_name());
      EXPECT_EQ(tp.name_.size(), 4);
      EXPECT_NE(tp.name_.data(), nullptr);
      EXPECT_EQ(tp.name_.data()[0], 't');
      EXPECT_EQ(tp.order_name(), 8);
      EXPECT_EQ(tp.name_, "test");
      EXPECT_EQ(tp2.name_.size(), 4);
      EXPECT_NE(tp2.name_.data(), nullptr);
      EXPECT_EQ(tp2.name_.data()[0], 't');
      EXPECT_EQ(tp2.order_name(), 8);
      EXPECT_EQ(tp2.name_, "test");
    } else {
      tp.name_.clear();
    }
    EXPECT_EQ(tp.name_.size(), 4);
    EXPECT_NE(tp.name_.data(), nullptr);
    EXPECT_EQ(tp.name_.data()[0], 't');
    EXPECT_EQ(tp.order_name(), 8);
    EXPECT_EQ(tp.name_, "test");
  }
}

TEST(onnx2_proto, TensorProtoNameStringToString2) {
  {
    TensorProto tp2;
    if (tp2.ref_name().size() == 0) {
      TensorProto tp;
      tp.name_ = "test";
      tp2.set_name(tp.ref_name());
      EXPECT_EQ(tp.name_.size(), 4);
      EXPECT_NE(tp.name_.data(), nullptr);
      EXPECT_EQ(tp.name_.data()[0], 't');
      EXPECT_EQ(tp.order_name(), 8);
      EXPECT_EQ(tp.name_, "test");
      EXPECT_EQ(tp2.name_.size(), 4);
      EXPECT_NE(tp2.name_.data(), nullptr);
      EXPECT_EQ(tp2.name_.data()[0], 't');
      EXPECT_EQ(tp2.order_name(), 8);
      EXPECT_EQ(tp2.name_, "test");
    } else {
      tp2.name_.clear();
    }
    EXPECT_EQ(tp2.name_.size(), 4);
    EXPECT_NE(tp2.name_.data(), nullptr);
    EXPECT_EQ(tp2.name_.data()[0], 't');
    EXPECT_EQ(tp2.order_name(), 8);
    EXPECT_EQ(tp2.name_, "test");
  }
}

TEST(onnx2_proto, TensorProtoName00) { TensorProto tp; }
TEST(onnx2_proto, TensorProtoName01) {
  TensorProto tp;
  tp.set_name("rt");
}

TEST(onnx2_proto, serialization_StringStringEntryProto) {
  StringStringEntryProto proto;
  proto.ref_key() = "key__";
  proto.ref_value() = "value__";
  EXPECT_EQ(proto.ref_key(), "key__");
  EXPECT_EQ(proto.ref_value(), "value__");
  std::string serie;
  proto.SerializeToString(serie);
  EXPECT_EQ(serie.size(), proto.SerializeSize());
  StringStringEntryProto proto2;
  proto2.ParseFromString(serie);
  EXPECT_EQ(proto.ref_key(), proto2.ref_key());
  EXPECT_EQ(proto.ref_value(), proto2.ref_value());
  std::string serie2;
  proto2.SerializeToString(serie2);
  EXPECT_EQ(serie2.size(), proto2.SerializeSize());
  EXPECT_EQ(serie, serie2);
}

TEST(onnx2_proto, serialization_StringStringEntryProto_Twice) {
  StringStringEntryProto proto;
  proto.set_key("key_");
  proto.set_value("value_");
  EXPECT_EQ(proto.ref_key(), "key_");
  EXPECT_EQ(proto.ref_value(), "value_");
  proto.set_key("key__");
  proto.set_value("value__");
  EXPECT_EQ(proto.ref_key(), "key__");
  EXPECT_EQ(proto.ref_value(), "value__");
  proto.ref_key() = "key___";
  proto.ref_value() = "value___";
  EXPECT_EQ(proto.ref_key(), "key___");
  EXPECT_EQ(proto.ref_value(), "value___");
}

TEST(onnx2_proto, TensorShapeProto1) {
  TensorShapeProto shape;
  TensorShapeProto::Dimension &dim = shape.add_dim();
  dim.set_dim_value(5);
  TensorShapeProto::Dimension &dim2 = shape.ref_dim().add();
  dim2.set_dim_param("dime");
  dim2.ref_denotation() = "jj";
  EXPECT_EQ(shape.ref_dim().size(), 2);
  EXPECT_EQ(shape.ref_dim()[0].ref_dim_value(), 5);
  EXPECT_EQ(shape.ref_dim()[0].ref_dim_param().size(), 0);
  EXPECT_EQ(shape.ref_dim()[1].ref_dim_param(), "dime");
  EXPECT_FALSE(shape.ref_dim()[1].has_dim_value());
  EXPECT_EQ(shape.ref_dim()[1].ref_denotation(), "jj");
}

TEST(onnx2_stream, ZigZagEncoding) {
  int64_t original_values[] = {0, -1, 1, -2, 2, INT64_MAX, INT64_MIN};

  for (auto val : original_values) {
    uint64_t encoded = utils::encodeZigZag64(val);
    int64_t decoded = utils::decodeZigZag64(encoded);
    EXPECT_EQ(decoded, val) << "ZigZag encoding/decoding failed for value: " << val;
  }
}

TEST(onnx2_stream, FieldNumber) {
  utils::FieldNumber fn;
  fn.field_number = 5;
  fn.wire_type = 2;

  std::string str = fn.string();
  EXPECT_FALSE(str.empty());
  EXPECT_NE(str.find("field_number=5"), std::string::npos);
  EXPECT_NE(str.find("wire_type=2"), std::string::npos);
}

class onnx2_stream_2 : public ::testing::Test {
protected:
  void SetUp() override {
    data = {0x96, 0x01,
            // int64_t
            0x2A,
            // int32_t
            0x18,
            // float: 3.14
            0xC3, 0xF5, 0x48, 0x40,
            // double: 2.71828
            0x4D, 0xFB, 0x21, 0x09, 0x40, 0x05, 0x5D, 0x40,
            // field number: 10, wire_type: 2 -> (10 << 3) | 2 = 82
            0x52,
            // string length: 5
            0x05,
            // string "hello"
            'h', 'e', 'l', 'l', 'o'};

    stream.Setup(data.data(), data.size());
  }

  std::vector<uint8_t> data;
  utils::StringStream stream;
};

TEST_F(onnx2_stream_2, NextUInt64) {
  uint64_t value = stream.next_uint64();
  EXPECT_EQ(value, 150);
}

TEST_F(onnx2_stream_2, NextInt64) {
  stream.next_uint64();

  int64_t value = stream.next_int64();
  EXPECT_EQ(value, 42);
}

TEST_F(onnx2_stream_2, NextInt32) {
  stream.next_uint64();
  stream.next_int64();

  int32_t value = stream.next_int32();
  EXPECT_EQ(value, 24);
}

TEST_F(onnx2_stream_2, NextFloat) {
  stream.next_uint64();
  stream.next_int64();
  stream.next_int32();
  float value = stream.next_float();
  EXPECT_NEAR(value, 3.14f, 0.0001f);
}

TEST_F(onnx2_stream_2, NextField) {
  stream.next_uint64();
  stream.next_int64();
  stream.next_int32();
  stream.next_float();
  stream.next_double();

  utils::FieldNumber field = stream.next_field();
  EXPECT_EQ(field.field_number, 10);
  EXPECT_EQ(field.wire_type, 2);
}

TEST_F(onnx2_stream_2, NextString) {
  stream.next_uint64();
  stream.next_int64();
  stream.next_int32();
  stream.next_float();
  stream.next_double();
  stream.next_field();

  utils::RefString value = stream.next_string();
  EXPECT_EQ(value.size(), 5);
  EXPECT_EQ(value, "hello");
}

TEST_F(onnx2_stream_2, ReadBytes) {
  const uint8_t* bytes = stream.read_bytes(2);
  EXPECT_EQ(bytes[0], 0x96);
  EXPECT_EQ(bytes[1], 0x01);
}

TEST_F(onnx2_stream_2, CanRead) {
  stream.CanRead(data.size(), "Test message");
  stream.read_bytes(10);
  stream.CanRead(data.size() - 10, "Test message");
  EXPECT_THROW(stream.CanRead(data.size(), "Test message"), std::runtime_error);
}

TEST_F(onnx2_stream_2, NotEnd) {
  EXPECT_TRUE(stream.NotEnd());
  stream.read_bytes(data.size() - 1);
  EXPECT_TRUE(stream.NotEnd());
  stream.read_bytes(1);
  EXPECT_FALSE(stream.NotEnd());
}

TEST_F(onnx2_stream_2, Tell) {
  EXPECT_EQ(stream.tell(), 0);

  stream.read_bytes(5);
  EXPECT_EQ(stream.tell(), 5);

  stream.read_bytes(10);
  EXPECT_EQ(stream.tell(), 15);
}

TEST(onnx2_stream, StringWriteStream) {
  utils::StringWriteStream stream;

  stream.write_variant_uint64(150);
  stream.write_int64(42);
  stream.write_int32(24);
  stream.write_float(3.14f);
  stream.write_double(2.71828);
  stream.write_field_header(10, 2);
  stream.write_string("hello");
  EXPECT_GT(stream.size(), 0);
  EXPECT_NE(stream.data(), nullptr);

  utils::StringStream readStream(stream.data(), stream.size());

  EXPECT_EQ(readStream.next_uint64(), 150);
  EXPECT_EQ(readStream.next_int64(), 42);
  EXPECT_EQ(readStream.next_int32(), 24);
  EXPECT_NEAR(readStream.next_float(), 3.14f, 0.0001f);
  EXPECT_NEAR(readStream.next_double(), 2.71828, 0.0001);

  utils::FieldNumber field = readStream.next_field();
  EXPECT_EQ(field.field_number, 10);
  EXPECT_EQ(field.wire_type, 2);

  utils::RefString str = readStream.next_string();
  EXPECT_EQ(str, "hello");
}

TEST(onnx2_stream, StringWriteStreamStrings) {
  utils::StringWriteStream stream;

  std::string stdStr = "standard string";
  stream.write_string(stdStr);
  utils::String str("custom string", 13);
  stream.write_string(str);
  utils::RefString refStr("reference string", 16);
  stream.write_string(refStr);
  utils::StringStream readStream(stream.data(), stream.size());

  utils::RefString read1 = readStream.next_string();
  EXPECT_EQ(read1, "standard string");

  utils::RefString read2 = readStream.next_string();
  EXPECT_EQ(read2, "custom string");

  utils::RefString read3 = readStream.next_string();
  EXPECT_EQ(read3, "reference string");
}

TEST(onnx2_stream, BorrowedWriteStream) {
  std::vector<uint8_t> data = {'h', 'e', 'l', 'l', 'o'};
  utils::BorrowedWriteStream stream(data.data(), data.size());
  EXPECT_EQ(stream.size(), 5);
  EXPECT_EQ(stream.data(), data.data());
  EXPECT_THROW(stream.write_raw_bytes(nullptr, 0), std::runtime_error);
}

TEST(onnx2_stream, NestedStringWriteStreams) {
  utils::StringWriteStream innerStream;

  innerStream.write_string("inner data");
  utils::StringWriteStream outerStream;
  outerStream.write_field_header(15, 2);

  outerStream.write_string_stream(innerStream);

  utils::StringStream readStream(outerStream.data(), outerStream.size());

  utils::FieldNumber field = readStream.next_field();
  EXPECT_EQ(field.field_number, 15);
  EXPECT_EQ(field.wire_type, 2);

  uint64_t length = readStream.next_uint64();
  readStream.LimitToNext(length);
  utils::RefString str = readStream.next_string();
  readStream.Restore();
  EXPECT_EQ(str, "inner data");
}

TEST(onnx2_stream, NextPackedElement) {
  std::vector<uint8_t> data = {// a float: 3.14
                               0xC3, 0xF5, 0x48, 0x40,
                               // int32: 42
                               0x2A, 0x00, 0x00, 0x00};

  utils::StringStream stream(data.data(), data.size());

  float f;
  stream.next_packed_element(f);
  EXPECT_NEAR(f, 3.14f, 0.0001f);

  int32_t i;
  stream.next_packed_element(i);
  EXPECT_EQ(i, 42);
}

TEST(onnx2_stream, ErrorCases) {
  std::vector<uint8_t> badData = {0x80, 0x80, 0x80};
  utils::StringStream badStream(badData.data(), badData.size());

  EXPECT_THROW(badStream.next_uint64(), std::runtime_error);

  std::vector<uint8_t> smallData = {0x01, 0x02};
  utils::StringStream smallStream(smallData.data(), smallData.size());

  EXPECT_THROW(smallStream.CanRead(3, "Test message"), std::runtime_error);
}

TEST(onnx2_proto, StringStringEntryProto_Basic) {
  StringStringEntryProto entry;

  EXPECT_TRUE(entry.ref_key().empty());
  EXPECT_TRUE(entry.ref_value().empty());
  EXPECT_FALSE(entry.has_key());
  EXPECT_FALSE(entry.has_value());

  entry.set_key("test_key");
  entry.set_value("test_value");

  EXPECT_EQ(entry.ref_key(), "test_key");
  EXPECT_EQ(entry.ref_value(), "test_value");
  EXPECT_TRUE(entry.has_key());
  EXPECT_TRUE(entry.has_value());

  EXPECT_EQ(entry.order_key(), 1);
  EXPECT_EQ(entry.order_value(), 2);
}

TEST(onnx2_proto, StringStringEntryProto_Serialization) {
  StringStringEntryProto entry;
  entry.set_key("test_key");
  entry.set_value("test_value");

  std::string serialized;
  entry.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), entry.SerializeSize());

  StringStringEntryProto entry2;
  entry2.ParseFromString(serialized);

  EXPECT_EQ(entry2.ref_key(), "test_key");
  EXPECT_EQ(entry2.ref_value(), "test_value");
}

TEST(onnx2_proto, IntIntListEntryProto_Basic) {
  IntIntListEntryProto entry;

  EXPECT_EQ(entry.ref_key(), 0);
  EXPECT_EQ(entry.ref_value().size(), 0);
  EXPECT_FALSE(entry.has_value());

  entry.set_key(42);
  entry.ref_value().push_back(1);
  entry.ref_value().push_back(2);
  entry.ref_value().push_back(3);

  EXPECT_EQ(entry.ref_key(), 42);
  EXPECT_EQ(entry.ref_value().size(), 3);
  EXPECT_EQ(entry.ref_value()[0], 1);
  EXPECT_EQ(entry.ref_value()[1], 2);
  EXPECT_EQ(entry.ref_value()[2], 3);
  EXPECT_TRUE(entry.has_key());
  EXPECT_TRUE(entry.has_value());

  EXPECT_EQ(entry.order_key(), 1);
  EXPECT_EQ(entry.order_value(), 2);
}

TEST(onnx2_proto, IntIntListEntryProto_Serialization) {
  IntIntListEntryProto entry;
  entry.set_key(42);
  entry.ref_value().push_back(1);
  entry.ref_value().push_back(2);
  entry.ref_value().push_back(3);

  std::string serialized;
  entry.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), entry.SerializeSize());

  IntIntListEntryProto entry2;
  entry2.ParseFromString(serialized);

  EXPECT_EQ(entry2.ref_key(), 42);
  EXPECT_EQ(entry2.ref_value().size(), 3);
  EXPECT_EQ(entry2.ref_value()[0], 1);
  EXPECT_EQ(entry2.ref_value()[1], 2);
  EXPECT_EQ(entry2.ref_value()[2], 3);
}

TEST(onnx2_proto, TensorAnnotation_Basic) {
  TensorAnnotation annotation;

  EXPECT_TRUE(annotation.ref_tensor_name().empty());
  EXPECT_EQ(annotation.ref_quant_parameter_tensor_names().size(), 0);

  annotation.set_tensor_name("my_tensor");
  StringStringEntryProto& entry = annotation.add_quant_parameter_tensor_names();
  entry.set_key("scale");
  entry.set_value("scale_tensor");

  EXPECT_EQ(annotation.ref_tensor_name(), "my_tensor");
  EXPECT_EQ(annotation.ref_quant_parameter_tensor_names().size(), 1);
  EXPECT_EQ(annotation.ref_quant_parameter_tensor_names()[0].ref_key(), "scale");
  EXPECT_EQ(annotation.ref_quant_parameter_tensor_names()[0].ref_value(), "scale_tensor");
}

TEST(onnx2_proto, DeviceConfigurationProto_Basic) {
  DeviceConfigurationProto config;

  EXPECT_TRUE(config.ref_name().empty());
  EXPECT_EQ(config.ref_num_devices(), 0);
  EXPECT_EQ(config.ref_device().size(), 0);

  config.set_name("CPU");
  config.set_num_devices(2);
  config.add_device() = "device0";
  config.add_device() = "device1";

  EXPECT_EQ(config.ref_name(), "CPU");
  EXPECT_EQ(config.ref_num_devices(), 2);
  EXPECT_EQ(config.ref_device().size(), 2);
  EXPECT_EQ(config.ref_device()[0], "device0");
  EXPECT_EQ(config.ref_device()[1], "device1");
}

TEST(onnx2_proto, SimpleShardedDimProto_Basic) {
  SimpleShardedDimProto dim;

  EXPECT_FALSE(dim.has_dim_value());
  EXPECT_TRUE(dim.ref_dim_param().empty());
  EXPECT_EQ(dim.ref_num_shards(), 0);

  dim.set_dim_value(100);
  dim.set_dim_param("batch");
  dim.set_num_shards(4);

  EXPECT_TRUE(dim.has_dim_value());
  EXPECT_EQ(dim.ref_dim_value(), 100);
  EXPECT_EQ(dim.ref_dim_param(), "batch");
  EXPECT_EQ(dim.ref_num_shards(), 4);
}

TEST(onnx2_proto, ShardedDimProto_Basic) {
  ShardedDimProto dim;

  EXPECT_EQ(dim.ref_axis(), 0);
  EXPECT_EQ(dim.ref_simple_sharding().size(), 0);

  dim.set_axis(1);
  SimpleShardedDimProto& simple_dim = dim.add_simple_sharding();
  simple_dim.set_dim_value(100);
  simple_dim.set_num_shards(4);

  EXPECT_EQ(dim.ref_axis(), 1);
  EXPECT_EQ(dim.ref_simple_sharding().size(), 1);
  EXPECT_EQ(dim.ref_simple_sharding()[0].ref_dim_value(), 100);
  EXPECT_EQ(dim.ref_simple_sharding()[0].ref_num_shards(), 4);
}

TEST(onnx2_proto, ShardingSpecProto_Basic) {
  ShardingSpecProto spec;

  EXPECT_TRUE(spec.ref_tensor_name().empty());
  EXPECT_EQ(spec.ref_device().size(), 0);
  EXPECT_EQ(spec.ref_index_to_device_group_map().size(), 0);
  EXPECT_EQ(spec.ref_sharded_dim().size(), 0);

  spec.set_tensor_name("my_tensor");

  spec.ref_device().push_back(0);
  spec.ref_device().push_back(1);

  IntIntListEntryProto& map_entry = spec.add_index_to_device_group_map();
  map_entry.set_key(0);
  map_entry.ref_value().push_back(0);

  ShardedDimProto& dim = spec.add_sharded_dim();
  dim.set_axis(0);

  EXPECT_EQ(spec.ref_tensor_name(), "my_tensor");
  EXPECT_EQ(spec.ref_device().size(), 2);
  EXPECT_EQ(spec.ref_device()[0], 0);
  EXPECT_EQ(spec.ref_device()[1], 1);
  EXPECT_EQ(spec.ref_index_to_device_group_map().size(), 1);
  EXPECT_EQ(spec.ref_index_to_device_group_map()[0].ref_key(), 0);
  EXPECT_EQ(spec.ref_sharded_dim().size(), 1);
  EXPECT_EQ(spec.ref_sharded_dim()[0].ref_axis(), 0);
}

TEST(onnx2_proto, NodeDeviceConfigurationProto_Basic) {
  NodeDeviceConfigurationProto config;

  EXPECT_TRUE(config.ref_configuration_id().empty());
  EXPECT_EQ(config.ref_sharding_spec().size(), 0);
  EXPECT_FALSE(config.has_pipeline_stage());

  config.set_configuration_id("config1");
  config.add_sharding_spec();
  config.set_pipeline_stage(2);

  EXPECT_EQ(config.ref_configuration_id(), "config1");
  EXPECT_EQ(config.ref_sharding_spec().size(), 1);
  EXPECT_TRUE(config.has_pipeline_stage());
  EXPECT_EQ(config.ref_pipeline_stage(), 2);
}

TEST(onnx2_proto, OperatorSetIdProto_Basic) {
  OperatorSetIdProto op_set;

  EXPECT_TRUE(op_set.ref_domain().empty());
  EXPECT_EQ(op_set.ref_version(), 0);

  op_set.set_domain("ai.onnx");
  op_set.set_version(12);

  EXPECT_EQ(op_set.ref_domain(), "ai.onnx");
  EXPECT_EQ(op_set.ref_version(), 12);
}

TEST(onnx2_proto, TensorShapeProto_Basic) {
  TensorShapeProto shape;

  EXPECT_EQ(shape.ref_dim().size(), 0);

  TensorShapeProto::Dimension &dim1 = shape.add_dim();
  dim1.set_dim_value(5);

  TensorShapeProto::Dimension &dim2 = shape.ref_dim().add();
  dim2.set_dim_param("N");
  dim2.set_denotation("batch");

  EXPECT_EQ(shape.ref_dim().size(), 2);
  EXPECT_TRUE(shape.ref_dim()[0].has_dim_value());
  EXPECT_EQ(shape.ref_dim()[0].ref_dim_value(), 5);
  EXPECT_FALSE(shape.ref_dim()[0].has_dim_param());

  EXPECT_FALSE(shape.ref_dim()[1].has_dim_value());
  EXPECT_EQ(shape.ref_dim()[1].ref_dim_param(), "N");
  EXPECT_EQ(shape.ref_dim()[1].ref_denotation(), "batch");
}

TEST(onnx2_proto, TensorShapeProto_Dimension) {
  TensorShapeProto::Dimension dim;

  EXPECT_FALSE(dim.has_dim_value());
  EXPECT_TRUE(dim.ref_dim_param().empty());
  EXPECT_TRUE(dim.ref_denotation().empty());

  dim.set_dim_value(10);
  EXPECT_TRUE(dim.has_dim_value());
  EXPECT_EQ(dim.ref_dim_value(), 10);

  dim.set_dim_param("batch_size");
  EXPECT_EQ(dim.ref_dim_param(), "batch_size");

  dim.set_denotation("batch");
  EXPECT_EQ(dim.ref_denotation(), "batch");
}

TEST(onnx2_proto, TensorProto_Basic) {
  TensorProto tensor;

  EXPECT_EQ(tensor.ref_data_type(), TensorProto::DataType::UNDEFINED);
  EXPECT_EQ(tensor.ref_dims().size(), 0);
  EXPECT_TRUE(tensor.ref_name().empty());

  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_dims().push_back(2);
  tensor.ref_dims().push_back(3);
  tensor.set_name("my_tensor");

  tensor.ref_float_data().push_back(1.0f);
  tensor.ref_float_data().push_back(2.0f);
  tensor.ref_float_data().push_back(3.0f);
  tensor.ref_float_data().push_back(4.0f);
  tensor.ref_float_data().push_back(5.0f);
  tensor.ref_float_data().push_back(6.0f);

  EXPECT_EQ(tensor.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor.ref_dims().size(), 2);
  EXPECT_EQ(tensor.ref_dims()[0], 2);
  EXPECT_EQ(tensor.ref_dims()[1], 3);
  EXPECT_EQ(tensor.ref_name(), "my_tensor");
  EXPECT_EQ(tensor.ref_float_data().size(), 6);
  EXPECT_EQ(tensor.ref_float_data()[0], 1.0f);
  EXPECT_EQ(tensor.ref_float_data()[5], 6.0f);
}

TEST(onnx2_proto, TensorProto_DataTypes) {
  TensorProto tensor;

  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_float_data().push_back(1.0f);
  tensor.ref_float_data().push_back(2.0f);
  EXPECT_EQ(tensor.ref_float_data().size(), 2);
  EXPECT_EQ(tensor.ref_float_data()[0], 1.0f);
  EXPECT_EQ(tensor.ref_float_data()[1], 2.0f);

  tensor.set_data_type(TensorProto::DataType::INT32);
  tensor.ref_int32_data().push_back(10);
  tensor.ref_int32_data().push_back(20);
  EXPECT_EQ(tensor.ref_int32_data().size(), 2);
  EXPECT_EQ(tensor.ref_int32_data()[0], 10);
  EXPECT_EQ(tensor.ref_int32_data()[1], 20);

  tensor.set_data_type(TensorProto::DataType::STRING);
  tensor.add_string_data() = "hello";
  tensor.add_string_data() = "world";
  EXPECT_EQ(tensor.ref_string_data().size(), 2);
  EXPECT_EQ(tensor.ref_string_data()[0], "hello");
  EXPECT_EQ(tensor.ref_string_data()[1], "world");

  tensor.set_data_type(TensorProto::DataType::INT64);
  tensor.ref_int64_data().push_back(100);
  tensor.ref_int64_data().push_back(200);
  EXPECT_EQ(tensor.ref_int64_data().size(), 2);
  EXPECT_EQ(tensor.ref_int64_data()[0], 100);
  EXPECT_EQ(tensor.ref_int64_data()[1], 200);

  tensor.set_data_type(TensorProto::DataType::DOUBLE);
  tensor.ref_double_data().push_back(1.5);
  tensor.ref_double_data().push_back(2.5);
  EXPECT_EQ(tensor.ref_double_data().size(), 2);
  EXPECT_EQ(tensor.ref_double_data()[0], 1.5);
  EXPECT_EQ(tensor.ref_double_data()[1], 2.5);

  tensor.set_data_type(TensorProto::DataType::UINT64);
  tensor.ref_uint64_data().push_back(1000);
  tensor.ref_uint64_data().push_back(2000);
  EXPECT_EQ(tensor.ref_uint64_data().size(), 2);
  EXPECT_EQ(tensor.ref_uint64_data()[0], 1000);
  EXPECT_EQ(tensor.ref_uint64_data()[1], 2000);
}

TEST(onnx2_proto, TensorProto_Segment) {
  TensorProto tensor;

  EXPECT_EQ(tensor.ref_segment().ref_begin(), 0);
  EXPECT_EQ(tensor.ref_segment().ref_end(), 0);

  tensor.ref_segment().set_begin(5);
  tensor.ref_segment().set_end(10);

  EXPECT_EQ(tensor.ref_segment().ref_begin(), 5);
  EXPECT_EQ(tensor.ref_segment().ref_end(), 10);
}

TEST(onnx2_proto, TensorProto_RawData) {
  TensorProto tensor;

  EXPECT_EQ(tensor.ref_raw_data().size(), 0);

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  tensor.ref_raw_data().resize(data.size() * sizeof(float));
  std::memcpy(tensor.ref_raw_data().data(), data.data(), data.size() * sizeof(float));

  EXPECT_EQ(tensor.ref_raw_data().size(), data.size() * sizeof(float));

  const float *raw_data_ptr = reinterpret_cast<const float*>(tensor.ref_raw_data().data());
  EXPECT_EQ(raw_data_ptr[0], 1.0f);
  EXPECT_EQ(raw_data_ptr[1], 2.0f);
  EXPECT_EQ(raw_data_ptr[2], 3.0f);
  EXPECT_EQ(raw_data_ptr[3], 4.0f);
}

TEST(onnx2_proto, TensorProto_Serialization) {
  TensorProto tensor1;
  tensor1.set_name("test_tensor");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.ref_dims().push_back(2);
  tensor1.ref_dims().push_back(2);
  tensor1.ref_float_data().push_back(1.0f);
  tensor1.ref_float_data().push_back(2.0f);
  tensor1.ref_float_data().push_back(3.0f);
  tensor1.ref_float_data().push_back(4.0f);

  std::string serialized;
  tensor1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), tensor1.SerializeSize());

  TensorProto tensor2;
  tensor2.ParseFromString(serialized);

  EXPECT_EQ(tensor2.ref_name(), "test_tensor");
  EXPECT_EQ(tensor2.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor2.ref_dims().size(), 2);
  EXPECT_EQ(tensor2.ref_dims()[0], 2);
  EXPECT_EQ(tensor2.ref_dims()[1], 2);
  EXPECT_EQ(tensor2.ref_float_data().size(), 4);
  EXPECT_EQ(tensor2.ref_float_data()[0], 1.0f);
  EXPECT_EQ(tensor2.ref_float_data()[1], 2.0f);
  EXPECT_EQ(tensor2.ref_float_data()[2], 3.0f);
  EXPECT_EQ(tensor2.ref_float_data()[3], 4.0f);
}

TEST(onnx2_proto, SparseTensorProto_Basic) {
  SparseTensorProto sparse;

  EXPECT_EQ(sparse.ref_dims().size(), 0);

  sparse.ref_dims().push_back(3);
  sparse.ref_dims().push_back(4);

  sparse.ref_values().set_data_type(TensorProto::DataType::FLOAT);
  sparse.ref_values().ref_float_data().push_back(5.0f);
  sparse.ref_values().ref_float_data().push_back(6.0f);

  sparse.ref_indices().set_data_type(TensorProto::DataType::INT64);
  sparse.ref_indices().ref_dims().push_back(2);
  sparse.ref_indices().ref_dims().push_back(2);
  sparse.ref_indices().ref_int64_data().push_back(0);
  sparse.ref_indices().ref_int64_data().push_back(2);
  sparse.ref_indices().ref_int64_data().push_back(1);
  sparse.ref_indices().ref_int64_data().push_back(3);

  EXPECT_EQ(sparse.ref_dims().size(), 2);
  EXPECT_EQ(sparse.ref_dims()[0], 3);
  EXPECT_EQ(sparse.ref_dims()[1], 4);

  EXPECT_EQ(sparse.ref_values().ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(sparse.ref_values().ref_float_data().size(), 2);
  EXPECT_EQ(sparse.ref_values().ref_float_data()[0], 5.0f);
  EXPECT_EQ(sparse.ref_values().ref_float_data()[1], 6.0f);

  EXPECT_EQ(sparse.ref_indices().ref_data_type(), TensorProto::DataType::INT64);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data().size(), 4);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[0], 0);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[1], 2);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[2], 1);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[3], 3);
}

TEST(onnx2_proto, TypeProto_Tensor) {
  TypeProto type;

  EXPECT_FALSE(type.has_tensor_type());

  type.add_tensor_type().set_elem_type(1); // FLOAT
  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_FALSE(type.ref_tensor_type().has_shape());
  TensorShapeProto& shape = type.ref_tensor_type().add_shape();
  EXPECT_TRUE(type.ref_tensor_type().has_shape());
  TensorShapeProto::Dimension &dim = shape.add_dim();
  dim.set_dim_value(3);

  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_EQ(type.ref_tensor_type().ref_elem_type(), 1);
  EXPECT_TRUE(type.ref_tensor_type().has_shape());
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim().size(), 1);
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 3);
}

TEST(onnx2_proto, CreateTensorProto) {
  TensorProto tensor;
  tensor.set_name("test_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_dims().push_back(2);
  tensor.ref_dims().push_back(3);

  for (int i = 0; i < 6; ++i) {
    tensor.ref_float_data().push_back(static_cast<float>(i + 1));
  }

  EXPECT_EQ(tensor.ref_name(), "test_tensor");
  EXPECT_EQ(tensor.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor.ref_dims().size(), 2);
  EXPECT_EQ(tensor.ref_dims()[0], 2);
  EXPECT_EQ(tensor.ref_dims()[1], 3);
  EXPECT_EQ(tensor.ref_float_data().size(), 6);
}

TEST(onnx2_proto, SerializeDeserializeTensorProto) {
  TensorProto tensor1;
  tensor1.set_name("serialized_tensor");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.ref_dims().push_back(2);
  tensor1.ref_dims().push_back(2);
  tensor1.ref_float_data().push_back(1.0f);
  tensor1.ref_float_data().push_back(2.0f);
  tensor1.ref_float_data().push_back(3.0f);
  tensor1.ref_float_data().push_back(4.0f);

  std::string serialized;
  tensor1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), tensor1.SerializeSize());

  TensorProto tensor2;
  tensor2.ParseFromString(serialized);

  EXPECT_EQ(tensor2.ref_name(), "serialized_tensor");
  EXPECT_EQ(tensor2.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor2.ref_dims().size(), 2);
  EXPECT_EQ(tensor2.ref_dims()[0], 2);
  EXPECT_EQ(tensor2.ref_dims()[1], 2);
  EXPECT_EQ(tensor2.ref_float_data().size(), 4);
  EXPECT_EQ(tensor2.ref_float_data()[0], 1.0f);
  EXPECT_EQ(tensor2.ref_float_data()[1], 2.0f);
  EXPECT_EQ(tensor2.ref_float_data()[2], 3.0f);
  EXPECT_EQ(tensor2.ref_float_data()[3], 4.0f);
}

TEST(onnx2_proto, TypeProtoOperations) {
  TypeProto type;

  type.add_tensor_type().set_elem_type(1); // FLOAT
  EXPECT_TRUE(type.has_tensor_type());

  TensorShapeProto& shape = type.ref_tensor_type().add_shape();

  TensorShapeProto::Dimension &dim1 = shape.add_dim();
  dim1.set_dim_value(3);

  TensorShapeProto::Dimension &dim2 = shape.add_dim();
  dim2.set_dim_param("batch_size");

  EXPECT_TRUE(type.has_tensor_type());
  EXPECT_EQ(type.ref_tensor_type().ref_elem_type(), 1);
  EXPECT_TRUE(type.ref_tensor_type().has_shape());
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim().size(), 2);
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 3);
  EXPECT_EQ(type.ref_tensor_type().ref_shape().ref_dim()[1].ref_dim_param(), "batch_size");
}

TEST(onnx2_proto, StringStringEntryProtoOperations) {
  StringStringEntryProto entry;
  entry.set_key("metadata_key");
  entry.set_value("metadata_value");

  EXPECT_EQ(entry.ref_key(), "metadata_key");
  EXPECT_EQ(entry.ref_value(), "metadata_value");

  std::string serialized;
  entry.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), entry.SerializeSize());

  StringStringEntryProto entry2;
  entry2.ParseFromString(serialized);

  EXPECT_EQ(entry2.ref_key(), "metadata_key");
  EXPECT_EQ(entry2.ref_value(), "metadata_value");
}

TEST(onnx2_proto, TensorProtoWithRawData) {
  TensorProto tensor;
  tensor.set_name("raw_data_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_dims().push_back(2);
  tensor.ref_dims().push_back(2);

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  tensor.ref_raw_data().resize(data.size() * sizeof(float));
  std::memcpy(tensor.ref_raw_data().data(), data.data(), data.size() * sizeof(float));

  EXPECT_EQ(tensor.ref_name(), "raw_data_tensor");
  EXPECT_EQ(tensor.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor.ref_dims().size(), 2);
  EXPECT_EQ(tensor.ref_dims()[0], 2);
  EXPECT_EQ(tensor.ref_dims()[1], 2);
  EXPECT_EQ(tensor.ref_raw_data().size(), data.size() * sizeof(float));

  const float *raw_data_ptr = reinterpret_cast<const float*>(tensor.ref_raw_data().data());
  EXPECT_EQ(raw_data_ptr[0], 1.0f);
  EXPECT_EQ(raw_data_ptr[1], 2.0f);
  EXPECT_EQ(raw_data_ptr[2], 3.0f);
  EXPECT_EQ(raw_data_ptr[3], 4.0f);
}

TEST(onnx2_proto, SparseTensorProtoOperations) {
  SparseTensorProto sparse;

  sparse.ref_dims().push_back(3);
  sparse.ref_dims().push_back(4);

  sparse.ref_values().set_data_type(TensorProto::DataType::FLOAT);
  sparse.ref_values().ref_float_data().push_back(5.0f);
  sparse.ref_values().ref_float_data().push_back(6.0f);

  sparse.ref_indices().set_data_type(TensorProto::DataType::INT64);
  sparse.ref_indices().ref_dims().push_back(2);
  sparse.ref_indices().ref_dims().push_back(2);
  sparse.ref_indices().ref_int64_data().push_back(0);
  sparse.ref_indices().ref_int64_data().push_back(2);
  sparse.ref_indices().ref_int64_data().push_back(1);
  sparse.ref_indices().ref_int64_data().push_back(3);

  EXPECT_EQ(sparse.ref_dims().size(), 2);
  EXPECT_EQ(sparse.ref_dims()[0], 3);
  EXPECT_EQ(sparse.ref_dims()[1], 4);

  EXPECT_EQ(sparse.ref_values().ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(sparse.ref_values().ref_float_data().size(), 2);
  EXPECT_EQ(sparse.ref_values().ref_float_data()[0], 5.0f);
  EXPECT_EQ(sparse.ref_values().ref_float_data()[1], 6.0f);

  EXPECT_EQ(sparse.ref_indices().ref_data_type(), TensorProto::DataType::INT64);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data().size(), 4);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[0], 0);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[1], 2);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[2], 1);
  EXPECT_EQ(sparse.ref_indices().ref_int64_data()[3], 3);

  std::string serialized;
  sparse.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), sparse.SerializeSize());

  SparseTensorProto sparse2;
  sparse2.ParseFromString(serialized);

  EXPECT_EQ(sparse2.ref_dims().size(), 2);
  EXPECT_EQ(sparse2.ref_values().ref_float_data().size(), 2);
  EXPECT_EQ(sparse2.ref_indices().ref_int64_data().size(), 4);
}

TEST(onnx2_proto, TensorShapeProtoOperations) {
  TensorShapeProto shape;

  TensorShapeProto::Dimension &dim1 = shape.add_dim();
  dim1.set_dim_value(5);

  TensorShapeProto::Dimension &dim2 = shape.add_dim();
  dim2.set_dim_param("N");
  dim2.set_denotation("batch");

  EXPECT_EQ(shape.ref_dim().size(), 2);
  EXPECT_TRUE(shape.ref_dim()[0].has_dim_value());
  EXPECT_EQ(shape.ref_dim()[0].ref_dim_value(), 5);
  EXPECT_FALSE(shape.ref_dim()[0].has_dim_param());

  EXPECT_FALSE(shape.ref_dim()[1].has_dim_value());
  EXPECT_EQ(shape.ref_dim()[1].ref_dim_param(), "N");
  EXPECT_EQ(shape.ref_dim()[1].ref_denotation(), "batch");

  std::string serialized;
  shape.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), shape.SerializeSize());

  TensorShapeProto shape2;
  shape2.ParseFromString(serialized);

  EXPECT_EQ(shape2.ref_dim().size(), 2);
  EXPECT_EQ(shape2.ref_dim()[0].ref_dim_value(), 5);
  EXPECT_EQ(shape2.ref_dim()[1].ref_dim_param(), "N");
  EXPECT_EQ(shape2.ref_dim()[1].ref_denotation(), "batch");
}

TEST(onnx2_proto, TensorProtoDataTypes) {
  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.ref_float_data().push_back(1.0f);
    tensor.ref_float_data().push_back(2.0f);
    EXPECT_EQ(tensor.ref_float_data().size(), 2);
    EXPECT_EQ(tensor.ref_float_data()[0], 1.0f);
    EXPECT_EQ(tensor.ref_float_data()[1], 2.0f);
  }

  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::INT32);
    tensor.ref_int32_data().push_back(10);
    tensor.ref_int32_data().push_back(20);
    EXPECT_EQ(tensor.ref_int32_data().size(), 2);
    EXPECT_EQ(tensor.ref_int32_data()[0], 10);
    EXPECT_EQ(tensor.ref_int32_data()[1], 20);
  }

  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::STRING);
    tensor.add_string_data() = "hello";
    tensor.add_string_data() = "world";
    EXPECT_EQ(tensor.ref_string_data().size(), 2);
    EXPECT_EQ(tensor.ref_string_data()[0], "hello");
    EXPECT_EQ(tensor.ref_string_data()[1], "world");
  }

  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::INT64);
    tensor.ref_int64_data().push_back(100);
    tensor.ref_int64_data().push_back(200);
    EXPECT_EQ(tensor.ref_int64_data().size(), 2);
    EXPECT_EQ(tensor.ref_int64_data()[0], 100);
    EXPECT_EQ(tensor.ref_int64_data()[1], 200);
  }

  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::DOUBLE);
    tensor.ref_double_data().push_back(1.5);
    tensor.ref_double_data().push_back(2.5);
    EXPECT_EQ(tensor.ref_double_data().size(), 2);
    EXPECT_EQ(tensor.ref_double_data()[0], 1.5);
    EXPECT_EQ(tensor.ref_double_data()[1], 2.5);
  }

  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::DataType::UINT64);
    tensor.ref_uint64_data().push_back(1000);
    tensor.ref_uint64_data().push_back(2000);
    EXPECT_EQ(tensor.ref_uint64_data().size(), 2);
    EXPECT_EQ(tensor.ref_uint64_data()[0], 1000);
    EXPECT_EQ(tensor.ref_uint64_data()[1], 2000);
  }
}

static TensorProto ToTensor(double value, TensorProto_DataType elem_type) {
  TensorProto t;
  t.set_data_type(elem_type);
  switch (elem_type) {
  case TensorProto_DataType::TensorProto_DataType_FLOAT:
    t.add_float_data((float)value);
    break;
  case TensorProto_DataType::TensorProto_DataType_DOUBLE:
    t.add_double_data(value);
    break;
  default:
    assert(false);
  }
  return t;
}

TEST(onnx2onnx, DataType) {
  TensorProto proto = ToTensor(4.5, TensorProto_DataType::TensorProto_DataType_FLOAT);
  EXPECT_EQ(proto.ref_float_data().size(), 1);
  EXPECT_EQ(proto.ref_float_data()[0], 4.5);
  EXPECT_EQ(proto.ref_data_type(), TensorProto_DataType::TensorProto_DataType_FLOAT);
}

TEST(onnx2_string, StringStringEntryProto) {
  utils::PrintOptions options;
  onnx2::StringStringEntryProto proto;
  proto.set_key("test_key");
  proto.set_value("test_value");
  std::vector<std::string> result = proto.PrintToVectorString(options);
  ASSERT_EQ(1, result.size());
  std::string serialized = result[0];
  EXPECT_TRUE(serialized.find("test_key") != std::string::npos);
  EXPECT_TRUE(serialized.find("test_value") != std::string::npos);
}

TEST(onnx2_string, IntIntListEntryProto) {
  utils::PrintOptions options;
  onnx2::IntIntListEntryProto proto;
  proto.set_key(42);
  proto.ref_value().push_back(1);
  proto.ref_value().push_back(2);
  proto.ref_value().push_back(3);
  std::vector<std::string> result = proto.PrintToVectorString(options);
  ASSERT_EQ(4, result.size());
  std::string serialized = utils::join_string(result, "\n");
  EXPECT_TRUE(serialized.find("42") != std::string::npos);
  EXPECT_TRUE(serialized.find("1") != std::string::npos);
  EXPECT_TRUE(serialized.find("2") != std::string::npos);
  EXPECT_TRUE(serialized.find("3") != std::string::npos);
}

TEST(onnx2_string, TensorAnnotation) {
  utils::PrintOptions options;
  onnx2::TensorAnnotation proto;
  proto.set_tensor_name("my_tensor");
  auto& entry = proto.add_quant_parameter_tensor_names();
  entry.set_key("scale");
  entry.set_value("scale_tensor");
  std::vector<std::string> result = proto.PrintToVectorString(options);
  ASSERT_EQ(6, result.size());
  std::string serialized = utils::join_string(result, "\n");
  EXPECT_TRUE(serialized.find("my_tensor") != std::string::npos);
  EXPECT_TRUE(serialized.find("scale") != std::string::npos);
  EXPECT_TRUE(serialized.find("scale_tensor") != std::string::npos);
}

TEST(onnx2_string, DeviceConfigurationProto) {
  utils::PrintOptions options;
  DeviceConfigurationProto config;
  config.set_name("test_device_config");
  config.set_num_devices(3);
  config.add_device() = "device1";
  config.add_device() = "device2";
  config.add_device() = "device3";

  std::vector<std::string> result = config.PrintToVectorString(options);

  ASSERT_FALSE(result.empty());

  bool foundName = false;
  bool foundNumDevices = false;
  bool foundDevices = false;

  std::string item = utils::join_string(result, "\n");
  if (item.find("name:") != std::string::npos && item.find("test_device_config") != std::string::npos) {
    foundName = true;
  }
  if (item.find("num_devices:") != std::string::npos && item.find("3") != std::string::npos) {
    foundNumDevices = true;
  }
  if (item.find("device:") != std::string::npos && item.find("device1") != std::string::npos &&
      item.find("device2") != std::string::npos && item.find("device3") != std::string::npos) {
    foundDevices = true;
  }

  EXPECT_TRUE(foundName);
  EXPECT_TRUE(foundNumDevices);
  EXPECT_TRUE(foundDevices);
}

TEST(onnx2_string, SimpleShardedDimProto) {
  utils::PrintOptions options;
  onnx2::SimpleShardedDimProto proto;
  proto.set_dim_value(100);
  proto.set_dim_param("batch_size");
  proto.set_num_shards(4);

  std::vector<std::string> result = proto.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundDimValue = false;
  bool foundDimParam = false;
  bool foundNumShards = false;

  for (const auto& item : result) {
    if (item.find("dim_value:") != std::string::npos && item.find("100") != std::string::npos) {
      foundDimValue = true;
    }
    if (item.find("dim_param:") != std::string::npos && item.find("batch_size") != std::string::npos) {
      foundDimParam = true;
    }
    if (item.find("num_shards:") != std::string::npos && item.find("4") != std::string::npos) {
      foundNumShards = true;
    }
  }

  EXPECT_TRUE(foundDimValue);
  EXPECT_TRUE(foundDimParam);
  EXPECT_TRUE(foundNumShards);
}

TEST(onnx2_string, ShardedDimProto) {
  utils::PrintOptions options;
  onnx2::ShardedDimProto proto;
  proto.set_axis(2);

  auto& simple_dim1 = proto.add_simple_sharding();
  simple_dim1.set_dim_value(100);
  simple_dim1.set_num_shards(4);

  auto& simple_dim2 = proto.add_simple_sharding();
  simple_dim2.set_dim_param("height");
  simple_dim2.set_num_shards(2);

  std::vector<std::string> result = proto.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundAxis = false;
  bool foundSimpleSharding = false;

  for (const auto& item : result) {
    if (item.find("axis:") != std::string::npos && item.find("2") != std::string::npos) {
      foundAxis = true;
    }
    if (item.find("simple_sharding") != std::string::npos) {
      foundSimpleSharding = true;
    }
  }

  EXPECT_TRUE(foundAxis);
  EXPECT_TRUE(foundSimpleSharding);
}

TEST(onnx2_string, ShardingSpecProto) {
  utils::PrintOptions options;
  onnx2::ShardingSpecProto proto;
  proto.set_tensor_name("sharded_tensor");

  proto.ref_device().push_back(0);
  proto.ref_device().push_back(1);
  proto.ref_device().push_back(2);

  auto& map_entry = proto.add_index_to_device_group_map();
  map_entry.set_key(0);
  map_entry.ref_value().push_back(0);
  map_entry.ref_value().push_back(1);

  auto& dim = proto.add_sharded_dim();
  dim.set_axis(1);
  auto& simple_dim = dim.add_simple_sharding();
  simple_dim.set_dim_value(64);
  simple_dim.set_num_shards(4);

  std::vector<std::string> result = proto.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundTensorName = false;
  bool foundDevice = false;
  bool foundMapping = false;
  bool foundShardedDim = false;

  for (const auto& item : result) {
    if (item.find("tensor_name:") != std::string::npos &&
        item.find("sharded_tensor") != std::string::npos) {
      foundTensorName = true;
    }
    if (item.find("device:") != std::string::npos) {
      foundDevice = true;
    }
    if (item.find("index_to_device_group_map") != std::string::npos) {
      foundMapping = true;
    }
    if (item.find("sharded_dim") != std::string::npos) {
      foundShardedDim = true;
    }
  }

  EXPECT_TRUE(foundTensorName);
  EXPECT_TRUE(foundDevice);
  EXPECT_TRUE(foundMapping);
  EXPECT_TRUE(foundShardedDim);
}

TEST(onnx2_string, NodeDeviceConfigurationProto) {
  utils::PrintOptions options;
  onnx2::NodeDeviceConfigurationProto proto;
  proto.set_configuration_id("node_config_1");
  proto.set_pipeline_stage(3);

  auto& spec = proto.add_sharding_spec();
  spec.set_tensor_name("input_tensor");
  spec.ref_device().push_back(0);
  spec.ref_device().push_back(1);

  std::vector<std::string> result = proto.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundConfigId = false;
  bool foundPipelineStage = false;
  bool foundShardingSpec = false;

  for (const auto& item : result) {
    if (item.find("configuration_id:") != std::string::npos &&
        item.find("node_config_1") != std::string::npos) {
      foundConfigId = true;
    }
    if (item.find("pipeline_stage:") != std::string::npos && item.find("3") != std::string::npos) {
      foundPipelineStage = true;
    }
    if (item.find("sharding_spec") != std::string::npos) {
      foundShardingSpec = true;
    }
  }

  EXPECT_TRUE(foundConfigId);
  EXPECT_TRUE(foundPipelineStage);
  EXPECT_TRUE(foundShardingSpec);
}

TEST(onnx2_string, OperatorSetIdProto) {
  utils::PrintOptions options;
  onnx2::OperatorSetIdProto proto;
  proto.set_domain("ai.onnx");
  proto.set_version(15);

  std::vector<std::string> result = proto.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundDomain = false;
  bool foundVersion = false;

  for (const auto& item : result) {
    if (item.find("domain:") != std::string::npos && item.find("ai.onnx") != std::string::npos) {
      foundDomain = true;
    }
    if (item.find("version:") != std::string::npos && item.find("15") != std::string::npos) {
      foundVersion = true;
    }
  }

  EXPECT_TRUE(foundDomain);
  EXPECT_TRUE(foundVersion);
}

TEST(onnx2_string, TensorShapeProto) {
  utils::PrintOptions options;
  onnx2::TensorShapeProto proto;

  auto& dim1 = proto.add_dim();
  dim1.set_dim_value(64);

  auto& dim2 = proto.add_dim();
  dim2.set_dim_param("batch");
  dim2.set_denotation("N");

  std::vector<std::string> result = proto.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundDim1 = false;
  bool foundDim2 = false;
  bool foundDenotation = false;

  std::string item = utils::join_string(result, "\n");
  if (item.find("dim") != std::string::npos && item.find("dim_value: 64") != std::string::npos) {
    foundDim1 = true;
  }
  if (item.find("dim_param: \"batch\"") != std::string::npos) {
    foundDim2 = true;
  }
  if (item.find("denotation: \"N\"") != std::string::npos) {
    foundDenotation = true;
  }

  EXPECT_TRUE(foundDim1);
  EXPECT_TRUE(foundDim2);
  EXPECT_TRUE(foundDenotation);
}

TEST(onnx2_string, TensorProto) {
  utils::PrintOptions options;
  onnx2::TensorProto proto;
  proto.set_name("test_tensor");
  proto.set_data_type(TensorProto::DataType::FLOAT);
  proto.ref_dims().push_back(3);
  proto.ref_dims().push_back(4);

  for (int i = 0; i < 12; ++i) {
    proto.ref_float_data().push_back(static_cast<float>(i * 0.5f));
  }

  proto.ref_doc_string() = "Un tenseur de test";

  std::vector<std::string> result = proto.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundName = false;
  bool foundDataType = false;
  bool foundDims = false;
  bool foundDocString = false;
  bool foundData = false;

  for (const auto& item : result) {
    if (item.find("name:") != std::string::npos && item.find("test_tensor") != std::string::npos) {
      foundName = true;
    }
    if (item.find("data_type:") != std::string::npos &&
        item.find(std::to_string(static_cast<int>(TensorProto::DataType::FLOAT))) !=
            std::string::npos) {
      foundDataType = true;
    }
    if (item.find("dims:") != std::string::npos) {
      foundDims = true;
    }
    if (item.find("doc_string:") != std::string::npos &&
        item.find("Un tenseur de test") != std::string::npos) {
      foundDocString = true;
    }
    if (item.find("float_data") != std::string::npos) {
      foundData = true;
    }
  }

  EXPECT_TRUE(foundName);
  EXPECT_TRUE(foundDataType);
  EXPECT_TRUE(foundDims);
  EXPECT_TRUE(foundDocString);
  EXPECT_TRUE(foundData);
}

TEST(onnx2_string, SparseTensorProto) {
  utils::PrintOptions options;
  onnx2::SparseTensorProto proto;

  proto.ref_dims().push_back(5);
  proto.ref_dims().push_back(5);

  proto.ref_values().set_name("values_tensor");
  proto.ref_values().set_data_type(TensorProto::DataType::FLOAT);
  proto.ref_values().ref_float_data().push_back(1.5f);
  proto.ref_values().ref_float_data().push_back(2.5f);
  proto.ref_values().ref_float_data().push_back(3.5f);

  proto.ref_indices().set_name("indices_tensor");
  proto.ref_indices().set_data_type(TensorProto::DataType::INT64);
  proto.ref_indices().ref_dims().push_back(3);
  proto.ref_indices().ref_dims().push_back(2);

  proto.ref_indices().ref_int64_data().push_back(0);
  proto.ref_indices().ref_int64_data().push_back(1);
  proto.ref_indices().ref_int64_data().push_back(2);
  proto.ref_indices().ref_int64_data().push_back(3);
  proto.ref_indices().ref_int64_data().push_back(4);
  proto.ref_indices().ref_int64_data().push_back(2);

  std::vector<std::string> result = proto.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundDims = false;
  bool foundValues = false;
  bool foundIndices = false;

  for (const auto& item : result) {
    if (item.find("dims:") != std::string::npos && item.find("5") != std::string::npos) {
      foundDims = true;
    }
    if (item.find("values") != std::string::npos && item.find("values_tensor") != std::string::npos) {
      foundValues = true;
    }
    if (item.find("indices") != std::string::npos && item.find("indices_tensor") != std::string::npos) {
      foundIndices = true;
    }
  }

  EXPECT_TRUE(foundDims);
  EXPECT_TRUE(foundValues);
  EXPECT_TRUE(foundIndices);
}

TEST(onnx2_string, TypeProto) {
  utils::PrintOptions options;
  onnx2::TypeProto proto;

  proto.add_tensor_type().set_elem_type(1); // FLOAT

  auto& shape = proto.ref_tensor_type().add_shape();

  auto& dim1 = shape.add_dim();
  dim1.set_dim_value(10);

  auto& dim2 = shape.add_dim();
  dim2.set_dim_param("batch");
  dim2.set_denotation("N");

  std::vector<std::string> result = proto.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundTensorType = false;
  bool foundElemType = false;
  bool foundShape = false;
  bool foundDimValue = false;
  bool foundDimParam = false;

  std::string item = utils::join_string(result, "\n");
  if (item.find("tensor_type") != std::string::npos) {
    foundTensorType = true;
  }
  if (item.find("elem_type: 1") != std::string::npos) {
    foundElemType = true;
  }
  if (item.find("shape") != std::string::npos) {
    foundShape = true;
  }
  if (item.find("dim_value: 10") != std::string::npos) {
    foundDimValue = true;
  }
  if (item.find("dim_param: \"batch\"") != std::string::npos) {
    foundDimParam = true;
  }

  EXPECT_TRUE(foundTensorType);
  EXPECT_TRUE(foundElemType);
  EXPECT_TRUE(foundShape);
  EXPECT_TRUE(foundDimValue);
  EXPECT_TRUE(foundDimParam);
}

TEST(onnx2_string, TensorProto_WithRawData) {
  utils::PrintOptions options;
  onnx2::TensorProto proto;
  proto.set_name("raw_data_tensor");
  proto.set_data_type(TensorProto::DataType::FLOAT);
  proto.ref_dims().push_back(2);
  proto.ref_dims().push_back(2);

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};

  proto.ref_raw_data().resize(data.size() * sizeof(float));
  std::memcpy(proto.ref_raw_data().data(), data.data(), data.size() * sizeof(float));

  std::vector<std::string> result = proto.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundName = false;
  bool foundDataType = false;
  bool foundRawData = false;

  for (const auto& item : result) {
    if (item.find("name:") != std::string::npos && item.find("raw_data_tensor") != std::string::npos) {
      foundName = true;
    }
    if (item.find("data_type:") != std::string::npos &&
        item.find(std::to_string(static_cast<int>(TensorProto::DataType::FLOAT))) !=
            std::string::npos) {
      foundDataType = true;
    }
    if (item.find("raw_data:") != std::string::npos) {
      foundRawData = true;
    }
  }

  EXPECT_TRUE(foundName);
  EXPECT_TRUE(foundDataType);
  EXPECT_TRUE(foundRawData);
}

TEST(onnx2_string, TensorProto_WithSegment) {
  utils::PrintOptions options;
  onnx2::TensorProto proto;
  proto.set_name("segmented_tensor");
  proto.set_data_type(TensorProto::DataType::FLOAT);

  proto.ref_segment().set_begin(5);
  proto.ref_segment().set_end(10);

  std::vector<std::string> result = proto.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundName = false;
  bool foundSegmentBegin = false;
  bool foundSegmentEnd = false;

  std::string item = utils::join_string(result, "\n");
  if (item.find("name:") != std::string::npos && item.find("segmented_tensor") != std::string::npos) {
    foundName = true;
  }
  if (item.find("segment") != std::string::npos && item.find("begin: 5") != std::string::npos) {
    foundSegmentBegin = true;
  }
  if (item.find("segment") != std::string::npos && item.find("end: 10") != std::string::npos) {
    foundSegmentEnd = true;
  }

  EXPECT_TRUE(foundName);
  EXPECT_TRUE(foundSegmentBegin);
  EXPECT_TRUE(foundSegmentEnd);
}

TEST(onnx2_proto, ValueInfoProto_Basic) {
  ValueInfoProto value_info;

  EXPECT_TRUE(value_info.ref_name().empty());
  EXPECT_TRUE(value_info.ref_doc_string().empty());
  EXPECT_FALSE(value_info.has_type());

  value_info.set_name("input_1");
  value_info.set_doc_string("Input tensor documentation");

  TypeProto& type = value_info.add_type();
  type.add_tensor_type().set_elem_type(1); // FLOAT
  TensorShapeProto& shape = type.ref_tensor_type().add_shape();
  TensorShapeProto::Dimension &dim = shape.add_dim();
  dim.set_dim_value(3);

  EXPECT_EQ(value_info.ref_name(), "input_1");
  EXPECT_EQ(value_info.ref_doc_string(), "Input tensor documentation");
  EXPECT_TRUE(value_info.has_type());
  EXPECT_TRUE(value_info.ref_type().has_tensor_type());
  EXPECT_EQ(value_info.ref_type().ref_tensor_type().ref_elem_type(), 1);
  EXPECT_TRUE(value_info.ref_type().ref_tensor_type().has_shape());
  EXPECT_EQ(value_info.ref_type().ref_tensor_type().ref_shape().ref_dim().size(), 1);
  EXPECT_EQ(value_info.ref_type().ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 3);
}

TEST(onnx2_proto, ValueInfoProto_Serialization) {
  ValueInfoProto value_info1;
  value_info1.set_name("output_1");
  value_info1.set_doc_string("Output tensor documentation");

  TypeProto& type = value_info1.add_type();
  type.add_tensor_type().set_elem_type(7); // INT64
  TensorShapeProto& shape = type.ref_tensor_type().add_shape();
  shape.add_dim().set_dim_value(2);
  shape.add_dim().set_dim_param("dynamic_dim");

  std::string serialized;
  value_info1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), value_info1.SerializeSize());

  ValueInfoProto value_info2;
  value_info2.ParseFromString(serialized);

  EXPECT_EQ(value_info2.ref_name(), "output_1");
  EXPECT_EQ(value_info2.ref_doc_string(), "Output tensor documentation");
  EXPECT_TRUE(value_info2.has_type());
  EXPECT_TRUE(value_info2.ref_type().has_tensor_type());
  EXPECT_EQ(value_info2.ref_type().ref_tensor_type().ref_elem_type(), 7);
  EXPECT_TRUE(value_info2.ref_type().ref_tensor_type().has_shape());
  EXPECT_EQ(value_info2.ref_type().ref_tensor_type().ref_shape().ref_dim().size(), 2);
  EXPECT_EQ(value_info2.ref_type().ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 2);
  EXPECT_EQ(value_info2.ref_type().ref_tensor_type().ref_shape().ref_dim()[1].ref_dim_param(),
            "dynamic_dim");
}

TEST(onnx2_proto, ValueInfoProto_PrintToVectorString) {
  utils::PrintOptions options;
  ValueInfoProto value_info;
  value_info.set_name("feature_vector");
  value_info.set_doc_string("Feature vector description");

  TypeProto& type = value_info.add_type();
  type.add_tensor_type().set_elem_type(1); // FLOAT
  TensorShapeProto& shape = type.ref_tensor_type().add_shape();
  shape.add_dim().set_dim_value(1);
  shape.add_dim().set_dim_value(512);

  std::vector<std::string> result = value_info.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundName = false;
  bool foundDocString = false;
  bool foundType = false;

  std::string serialized = utils::join_string(result, "\n");
  if (serialized.find("name:") != std::string::npos &&
      serialized.find("feature_vector") != std::string::npos) {
    foundName = true;
  }
  if (serialized.find("doc_string:") != std::string::npos &&
      serialized.find("Feature vector description") != std::string::npos) {
    foundDocString = true;
  }
  if (serialized.find("type") != std::string::npos &&
      serialized.find("elem_type: 1") != std::string::npos) {
    foundType = true;
  }

  EXPECT_TRUE(foundName);
  EXPECT_TRUE(foundDocString);
  EXPECT_TRUE(foundType);
}

TEST(onnx2_proto, CopyFrom_TensorProto) {
  TensorProto source;
  source.set_name("source_tensor");
  source.set_data_type(TensorProto::DataType::FLOAT);
  source.ref_dims().push_back(2);
  source.ref_dims().push_back(3);
  source.ref_float_data().push_back(1.0f);
  source.ref_float_data().push_back(2.0f);
  source.ref_float_data().push_back(3.0f);
  source.ref_raw_data().resize(12);
  source.set_doc_string("Source tensor documentation");

  TensorProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_name(), "source_tensor");
  EXPECT_EQ(target.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(target.ref_dims().size(), 2);
  EXPECT_EQ(target.ref_dims()[0], 2);
  EXPECT_EQ(target.ref_dims()[1], 3);
  EXPECT_EQ(target.ref_float_data().size(), 3);
  EXPECT_EQ(target.ref_float_data()[0], 1.0f);
  EXPECT_EQ(target.ref_float_data()[1], 2.0f);
  EXPECT_EQ(target.ref_float_data()[2], 3.0f);
  EXPECT_EQ(target.ref_raw_data().size(), 12);
  EXPECT_EQ(target.ref_doc_string(), "Source tensor documentation");
}

TEST(onnx2_proto, CopyFrom_ValueInfoProto) {
  ValueInfoProto source;
  source.set_name("source_info");
  source.set_doc_string("Source documentation");
  TypeProto& type = source.add_type();
  type.add_tensor_type().set_elem_type(1);

  ValueInfoProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_name(), "source_info");
  EXPECT_EQ(target.ref_doc_string(), "Source documentation");
  EXPECT_TRUE(target.has_type());
  EXPECT_TRUE(target.ref_type().has_tensor_type());
  EXPECT_EQ(target.ref_type().ref_tensor_type().ref_elem_type(), 1);
}

TEST(onnx2_proto, CopyFrom_TypeProto) {
  TypeProto source;
  source.add_tensor_type().set_elem_type(7);
  TensorShapeProto& shape = source.ref_tensor_type().add_shape();
  shape.add_dim().set_dim_value(10);
  shape.add_dim().set_dim_param("N");

  TypeProto target;
  target.CopyFrom(source);

  EXPECT_TRUE(target.has_tensor_type());
  EXPECT_EQ(target.ref_tensor_type().ref_elem_type(), 7);
  EXPECT_TRUE(target.ref_tensor_type().has_shape());
  EXPECT_EQ(target.ref_tensor_type().ref_shape().ref_dim().size(), 2);
  EXPECT_EQ(target.ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 10);
  EXPECT_EQ(target.ref_tensor_type().ref_shape().ref_dim()[1].ref_dim_param(), "N");
}

TEST(onnx2_proto, CopyFrom_SparseTensorProto) {
  SparseTensorProto source;
  source.ref_dims().push_back(4);
  source.ref_dims().push_back(4);

  source.ref_indices().set_name("indices");
  source.ref_indices().set_data_type(TensorProto::DataType::INT64);
  source.ref_indices().ref_int64_data().push_back(0);
  source.ref_indices().ref_int64_data().push_back(1);

  source.ref_values().set_name("values");
  source.ref_values().set_data_type(TensorProto::DataType::FLOAT);
  source.ref_values().ref_float_data().push_back(1.5f);

  SparseTensorProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_dims().size(), 2);
  EXPECT_EQ(target.ref_dims()[0], 4);
  EXPECT_EQ(target.ref_dims()[1], 4);
  EXPECT_EQ(target.ref_indices().ref_name(), "indices");
  EXPECT_EQ(target.ref_indices().ref_data_type(), TensorProto::DataType::INT64);
  EXPECT_EQ(target.ref_indices().ref_int64_data().size(), 2);
  EXPECT_EQ(target.ref_values().ref_name(), "values");
  EXPECT_EQ(target.ref_values().ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(target.ref_values().ref_float_data().size(), 1);
  EXPECT_EQ(target.ref_values().ref_float_data()[0], 1.5f);
}

TEST(onnx2_proto, AttributeProto_Basic) {
  AttributeProto attribute;

  EXPECT_TRUE(attribute.ref_name().empty());
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::UNDEFINED);
  EXPECT_FALSE(attribute.has_i());
  EXPECT_FALSE(attribute.has_f());
  EXPECT_FALSE(attribute.has_s());
  EXPECT_EQ(attribute.ref_ints().size(), 0);
  EXPECT_EQ(attribute.ref_floats().size(), 0);
  EXPECT_EQ(attribute.ref_strings().size(), 0);

  attribute.set_name("weight_decay");
  attribute.set_type(AttributeProto::AttributeType::FLOAT);
  attribute.set_f(0.01f);

  EXPECT_EQ(attribute.ref_name(), "weight_decay");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::FLOAT);
  EXPECT_TRUE(attribute.has_f());
  EXPECT_EQ(attribute.ref_f(), 0.01f);
}

TEST(onnx2_proto, AttributeProto_IntAttribute) {
  AttributeProto attribute;

  attribute.set_name("axis");
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(2);

  EXPECT_EQ(attribute.ref_name(), "axis");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::INT);
  EXPECT_TRUE(attribute.has_i());
  EXPECT_EQ(attribute.ref_i(), 2);
  EXPECT_FALSE(attribute.has_f());
  EXPECT_FALSE(attribute.has_s());
}

TEST(onnx2_proto, AttributeProto_StringAttribute) {
  AttributeProto attribute;

  attribute.set_name("mode");
  attribute.set_type(AttributeProto::AttributeType::STRING);
  attribute.set_s("constant");

  EXPECT_EQ(attribute.ref_name(), "mode");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::STRING);
  EXPECT_TRUE(attribute.has_s());
  EXPECT_EQ(attribute.ref_s(), "constant");
  EXPECT_FALSE(attribute.has_i());
  EXPECT_FALSE(attribute.has_f());
}

TEST(onnx2_proto, AttributeProto_IntsAttribute) {
  AttributeProto attribute;

  attribute.set_name("pads");
  attribute.set_type(AttributeProto::AttributeType::INTS);
  attribute.ref_ints().push_back(0);
  attribute.ref_ints().push_back(0);
  attribute.ref_ints().push_back(1);
  attribute.ref_ints().push_back(1);

  EXPECT_EQ(attribute.ref_name(), "pads");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::INTS);
  EXPECT_EQ(attribute.ref_ints().size(), 4);
  EXPECT_EQ(attribute.ref_ints()[0], 0);
  EXPECT_EQ(attribute.ref_ints()[1], 0);
  EXPECT_EQ(attribute.ref_ints()[2], 1);
  EXPECT_EQ(attribute.ref_ints()[3], 1);
}

TEST(onnx2_proto, AttributeProto_FloatsAttribute) {
  AttributeProto attribute;

  attribute.set_name("scales");
  attribute.set_type(AttributeProto::AttributeType::FLOATS);
  attribute.ref_floats().push_back(1.0f);
  attribute.ref_floats().push_back(2.0f);
  attribute.ref_floats().push_back(3.0f);

  EXPECT_EQ(attribute.ref_name(), "scales");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::FLOATS);
  EXPECT_EQ(attribute.ref_floats().size(), 3);
  EXPECT_EQ(attribute.ref_floats()[0], 1.0f);
  EXPECT_EQ(attribute.ref_floats()[1], 2.0f);
  EXPECT_EQ(attribute.ref_floats()[2], 3.0f);
}

TEST(onnx2_proto, AttributeProto_StringsAttribute) {
  AttributeProto attribute;

  attribute.set_name("tags");
  attribute.set_type(AttributeProto::AttributeType::STRINGS);
  attribute.add_strings() = "tag1";
  attribute.add_strings() = "tag2";
  attribute.add_strings() = "tag3";

  EXPECT_EQ(attribute.ref_name(), "tags");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::STRINGS);
  EXPECT_EQ(attribute.ref_strings().size(), 3);
  EXPECT_EQ(attribute.ref_strings()[0], "tag1");
  EXPECT_EQ(attribute.ref_strings()[1], "tag2");
  EXPECT_EQ(attribute.ref_strings()[2], "tag3");
}

TEST(onnx2_proto, AttributeProto_TensorAttribute) {
  AttributeProto attribute;

  attribute.set_name("value");
  attribute.set_type(AttributeProto::AttributeType::TENSOR);

  TensorProto& tensor = attribute.add_t();
  tensor.set_name("const_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_dims().push_back(2);
  tensor.ref_dims().push_back(2);
  tensor.ref_float_data().push_back(1.0f);
  tensor.ref_float_data().push_back(2.0f);
  tensor.ref_float_data().push_back(3.0f);
  tensor.ref_float_data().push_back(4.0f);

  EXPECT_EQ(attribute.ref_name(), "value");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::TENSOR);
  EXPECT_TRUE(attribute.has_t());
  EXPECT_EQ(attribute.ref_t().ref_name(), "const_tensor");
  EXPECT_EQ(attribute.ref_t().ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(attribute.ref_t().ref_dims().size(), 2);
  EXPECT_EQ(attribute.ref_t().ref_float_data().size(), 4);
}

TEST(onnx2_proto, AttributeProto_Serialization) {
  AttributeProto attribute;
  attribute.set_name("test_attribute");
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(42);
  attribute.set_doc_string("Test attribute documentation");

  std::string serialized;
  attribute.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), attribute.SerializeSize());

  AttributeProto attribute2;
  attribute2.ParseFromString(serialized);

  EXPECT_EQ(attribute2.ref_name(), "test_attribute");
  EXPECT_EQ(attribute2.ref_type(), AttributeProto::AttributeType::INT);
  EXPECT_EQ(attribute2.ref_i(), 42);
  EXPECT_EQ(attribute2.ref_doc_string(), "Test attribute documentation");
}

TEST(onnx2_string, AttributeProto) {
  utils::PrintOptions options;
  AttributeProto attribute;
  attribute.set_name("dropout_ratio");
  attribute.set_type(AttributeProto::AttributeType::FLOAT);
  attribute.set_f(0.5f);
  attribute.set_doc_string("Dropout ratio documentation");

  std::vector<std::string> result = attribute.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundName = false;
  bool foundValue = false;

  std::string serialized = utils::join_string(result, "\n");
  if (serialized.find("dropout_ratio") != std::string::npos) {
    foundName = true;
  }
  if (serialized.find(": 0.5") != std::string::npos) {
    foundValue = true;
  }
  EXPECT_TRUE(foundName);
  EXPECT_TRUE(foundValue);
}

TEST(onnx2_proto, AttributeProto_CopyFrom) {
  AttributeProto source;
  source.set_name("source_attribute");
  source.set_type(AttributeProto::AttributeType::INTS);
  source.ref_ints().push_back(10);
  source.ref_ints().push_back(20);
  source.set_doc_string("Source documentation");

  AttributeProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_name(), "source_attribute");
  EXPECT_EQ(target.ref_type(), AttributeProto::AttributeType::INTS);
  EXPECT_EQ(target.ref_ints().size(), 2);
  EXPECT_EQ(target.ref_ints()[0], 10);
  EXPECT_EQ(target.ref_ints()[1], 20);
  EXPECT_EQ(target.ref_doc_string(), "Source documentation");
}

TEST(onnx2_proto, AttributeProto_GraphAttribute) {
  AttributeProto attribute;

  attribute.set_name("body");
  attribute.set_type(AttributeProto::AttributeType::GRAPH);

  // Assuming GraphProto has methods similar to TensorProto
  attribute.add_g().set_name("subgraph");

  EXPECT_EQ(attribute.ref_name(), "body");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::GRAPH);
  EXPECT_TRUE(attribute.has_g());
  EXPECT_EQ(attribute.ref_g().ref_name(), "subgraph");
}

// NodeProto

TEST(onnx2_proto, NodeProto_Basic) {
  NodeProto node;

  EXPECT_TRUE(node.ref_name().empty());
  EXPECT_TRUE(node.ref_op_type().empty());
  EXPECT_TRUE(node.ref_domain().empty());
  EXPECT_EQ(node.ref_input().size(), 0);
  EXPECT_EQ(node.ref_output().size(), 0);
  EXPECT_EQ(node.ref_attribute().size(), 0);
  EXPECT_TRUE(node.ref_doc_string().empty());

  node.set_name("test_node");
  node.set_op_type("Conv");
  node.set_domain("ai.onnx");
  node.set_doc_string("Test node documentation");

  EXPECT_EQ(node.ref_name(), "test_node");
  EXPECT_EQ(node.ref_op_type(), "Conv");
  EXPECT_EQ(node.ref_domain(), "ai.onnx");
  EXPECT_EQ(node.ref_doc_string(), "Test node documentation");
}

TEST(onnx2_proto, NodeProto_InputOutput) {
  NodeProto node;

  EXPECT_EQ(node.ref_input().size(), 0);
  EXPECT_EQ(node.ref_output().size(), 0);

  // Add inputs
  node.add_input() = "X";
  node.add_input() = "W";
  node.add_input() = "B";

  // Add outputs
  node.add_output() = "Y";

  EXPECT_EQ(node.ref_input().size(), 3);
  EXPECT_EQ(node.ref_input()[0], "X");
  EXPECT_EQ(node.ref_input()[1], "W");
  EXPECT_EQ(node.ref_input()[2], "B");

  EXPECT_EQ(node.ref_output().size(), 1);
  EXPECT_EQ(node.ref_output()[0], "Y");

  // Test clear_input and clear_output
  node.clr_input();
  EXPECT_EQ(node.ref_input().size(), 0);

  node.clr_output();
  EXPECT_EQ(node.ref_output().size(), 0);
}

TEST(onnx2_proto, NodeProto_Attributes) {
  NodeProto node;

  EXPECT_EQ(node.ref_attribute().size(), 0);

  // Add attributes
  AttributeProto& attr1 = node.add_attribute();
  attr1.set_name("kernel_shape");
  attr1.set_type(AttributeProto::AttributeType::INTS);
  attr1.ref_ints().push_back(3);
  attr1.ref_ints().push_back(3);

  AttributeProto& attr2 = node.add_attribute();
  attr2.set_name("strides");
  attr2.set_type(AttributeProto::AttributeType::INTS);
  attr2.ref_ints().push_back(1);
  attr2.ref_ints().push_back(1);

  AttributeProto& attr3 = node.add_attribute();
  attr3.set_name("pads");
  attr3.set_type(AttributeProto::AttributeType::INTS);
  attr3.ref_ints().push_back(1);
  attr3.ref_ints().push_back(1);
  attr3.ref_ints().push_back(1);
  attr3.ref_ints().push_back(1);

  EXPECT_EQ(node.ref_attribute().size(), 3);
  EXPECT_EQ(node.ref_attribute()[0].ref_name(), "kernel_shape");
  EXPECT_EQ(node.ref_attribute()[0].ref_ints().size(), 2);
  EXPECT_EQ(node.ref_attribute()[1].ref_name(), "strides");
  EXPECT_EQ(node.ref_attribute()[1].ref_ints().size(), 2);
  EXPECT_EQ(node.ref_attribute()[2].ref_name(), "pads");
  EXPECT_EQ(node.ref_attribute()[2].ref_ints().size(), 4);
}

TEST(onnx2_proto, NodeProto_Serialization) {
  NodeProto node1;
  node1.set_name("conv1");
  node1.set_op_type("Conv");
  node1.set_domain("ai.onnx");

  node1.add_input() = "X";
  node1.add_input() = "W";

  node1.add_output() = "Y";

  AttributeProto& attr = node1.add_attribute();
  attr.set_name("kernel_shape");
  attr.set_type(AttributeProto::AttributeType::INTS);
  attr.ref_ints().push_back(3);
  attr.ref_ints().push_back(3);

  std::string serialized;
  node1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), node1.SerializeSize());

  NodeProto node2;
  node2.ParseFromString(serialized);

  EXPECT_EQ(node2.ref_name(), "conv1");
  EXPECT_EQ(node2.ref_op_type(), "Conv");
  EXPECT_EQ(node2.ref_domain(), "ai.onnx");
  EXPECT_EQ(node2.ref_input().size(), 2);
  EXPECT_EQ(node2.ref_input()[0], "X");
  EXPECT_EQ(node2.ref_input()[1], "W");
  EXPECT_EQ(node2.ref_output().size(), 1);
  EXPECT_EQ(node2.ref_output()[0], "Y");
  EXPECT_EQ(node2.ref_attribute().size(), 1);
  EXPECT_EQ(node2.ref_attribute()[0].ref_name(), "kernel_shape");
  EXPECT_EQ(node2.ref_attribute()[0].ref_ints().size(), 2);
  EXPECT_EQ(node2.ref_attribute()[0].ref_ints()[0], 3);
  EXPECT_EQ(node2.ref_attribute()[0].ref_ints()[1], 3);
}

TEST(onnx2_string, NodeProto_PrintToVectorString) {
  utils::PrintOptions options;
  NodeProto node;
  node.set_name("relu1");
  node.set_op_type("Relu");
  node.add_input() = "X";
  node.add_output() = "Y";
  node.set_doc_string("Simple ReLU activation");

  std::vector<std::string> result = node.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundName = false;
  bool foundOpType = false;
  bool foundInput = false;
  bool foundOutput = false;
  bool foundDocString = false;

  std::string serialized = utils::join_string(result, "\n");

  if (serialized.find("name:") != std::string::npos && serialized.find("relu1") != std::string::npos) {
    foundName = true;
  }

  if (serialized.find("op_type:") != std::string::npos &&
      serialized.find("Relu") != std::string::npos) {
    foundOpType = true;
  }

  if (serialized.find("input:") != std::string::npos && serialized.find("X") != std::string::npos) {
    foundInput = true;
  }

  if (serialized.find("output:") != std::string::npos && serialized.find("Y") != std::string::npos) {
    foundOutput = true;
  }

  if (serialized.find("doc_string:") != std::string::npos &&
      serialized.find("Simple ReLU activation") != std::string::npos) {
    foundDocString = true;
  }

  EXPECT_TRUE(foundName);
  EXPECT_TRUE(foundOpType);
  EXPECT_TRUE(foundInput);
  EXPECT_TRUE(foundOutput);
  EXPECT_TRUE(foundDocString);
}

TEST(onnx2_proto, NodeProto_CopyFrom) {
  NodeProto source;
  source.set_name("source_node");
  source.set_op_type("Add");
  source.set_domain("ai.onnx");
  source.add_input() = "A";
  source.add_input() = "B";
  source.add_output() = "C";

  AttributeProto& attr = source.add_attribute();
  attr.set_name("axis");
  attr.set_type(AttributeProto::AttributeType::INT);
  attr.set_i(1);

  source.set_doc_string("Source node documentation");

  NodeProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_name(), "source_node");
  EXPECT_EQ(target.ref_op_type(), "Add");
  EXPECT_EQ(target.ref_domain(), "ai.onnx");
  EXPECT_EQ(target.ref_input().size(), 2);
  EXPECT_EQ(target.ref_input()[0], "A");
  EXPECT_EQ(target.ref_input()[1], "B");
  EXPECT_EQ(target.ref_output().size(), 1);
  EXPECT_EQ(target.ref_output()[0], "C");
  EXPECT_EQ(target.ref_attribute().size(), 1);
  EXPECT_EQ(target.ref_attribute()[0].ref_name(), "axis");
  EXPECT_EQ(target.ref_attribute()[0].ref_i(), 1);
  EXPECT_EQ(target.ref_doc_string(), "Source node documentation");
}

TEST(onnx2_proto, NodeProto_ComplexModel) {
  // Create a more complex node to test multiple attributes, inputs, outputs
  NodeProto node;
  node.set_name("gemm1");
  node.set_op_type("Gemm");
  node.set_domain("ai.onnx");

  // Add inputs
  node.add_input() = "A";
  node.add_input() = "B";
  node.add_input() = "C";

  // Add outputs
  node.add_output() = "Y";

  // Add attributes
  AttributeProto& alpha = node.add_attribute();
  alpha.set_name("alpha");
  alpha.set_type(AttributeProto::AttributeType::FLOAT);
  alpha.set_f(0.5f);

  AttributeProto& beta = node.add_attribute();
  beta.set_name("beta");
  beta.set_type(AttributeProto::AttributeType::FLOAT);
  beta.set_f(0.8f);

  AttributeProto& transA = node.add_attribute();
  transA.set_name("transA");
  transA.set_type(AttributeProto::AttributeType::INT);
  transA.set_i(1);

  AttributeProto& transB = node.add_attribute();
  transB.set_name("transB");
  transB.set_type(AttributeProto::AttributeType::INT);
  transB.set_i(0);

  node.set_doc_string("GEMM operation: Y = alpha * A' * B + beta * C");

  EXPECT_EQ(node.ref_name(), "gemm1");
  EXPECT_EQ(node.ref_op_type(), "Gemm");
  EXPECT_EQ(node.ref_domain(), "ai.onnx");

  EXPECT_EQ(node.ref_input().size(), 3);
  EXPECT_EQ(node.ref_input()[0], "A");
  EXPECT_EQ(node.ref_input()[1], "B");
  EXPECT_EQ(node.ref_input()[2], "C");

  EXPECT_EQ(node.ref_output().size(), 1);
  EXPECT_EQ(node.ref_output()[0], "Y");

  EXPECT_EQ(node.ref_attribute().size(), 4);
  EXPECT_EQ(node.ref_attribute()[0].ref_name(), "alpha");
  EXPECT_EQ(node.ref_attribute()[0].ref_f(), 0.5f);
  EXPECT_EQ(node.ref_attribute()[1].ref_name(), "beta");
  EXPECT_EQ(node.ref_attribute()[1].ref_f(), 0.8f);
  EXPECT_EQ(node.ref_attribute()[2].ref_name(), "transA");
  EXPECT_EQ(node.ref_attribute()[2].ref_i(), 1);
  EXPECT_EQ(node.ref_attribute()[3].ref_name(), "transB");
  EXPECT_EQ(node.ref_attribute()[3].ref_i(), 0);

  EXPECT_EQ(node.ref_doc_string(), "GEMM operation: Y = alpha * A' * B + beta * C");
}

TEST(onnx2_proto, NodeProto_EmptyStrings) {
  NodeProto node;
  node.set_name("");
  node.set_op_type("Identity");
  node.add_input() = "";
  node.add_output() = "";

  EXPECT_TRUE(node.ref_name().empty());
  EXPECT_EQ(node.ref_op_type(), "Identity");
  EXPECT_EQ(node.ref_input().size(), 1);
  EXPECT_TRUE(node.ref_input()[0].empty());
  EXPECT_EQ(node.ref_output().size(), 1);
  EXPECT_TRUE(node.ref_output()[0].empty());

  std::string serialized;
  node.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), node.SerializeSize());

  NodeProto node2;
  node2.ParseFromString(serialized);

  EXPECT_TRUE(node2.ref_name().empty());
  EXPECT_EQ(node2.ref_op_type(), "Identity");
  EXPECT_EQ(node2.ref_input().size(), 1);
  EXPECT_TRUE(node2.ref_input()[0].empty());
  EXPECT_EQ(node2.ref_output().size(), 1);
  EXPECT_TRUE(node2.ref_output()[0].empty());
}

// GraphProto

TEST(onnx2_proto, GraphProto_Basic) {
  GraphProto graph;

  EXPECT_TRUE(graph.ref_name().empty());
  EXPECT_EQ(graph.ref_node().size(), 0);
  EXPECT_EQ(graph.ref_initializer().size(), 0);
  EXPECT_EQ(graph.ref_input().size(), 0);
  EXPECT_EQ(graph.ref_output().size(), 0);
  EXPECT_EQ(graph.ref_value_info().size(), 0);
  EXPECT_TRUE(graph.ref_doc_string().empty());

  graph.set_name("test_graph");
  graph.set_doc_string("Test graph documentation");

  EXPECT_EQ(graph.ref_name(), "test_graph");
  EXPECT_EQ(graph.ref_doc_string(), "Test graph documentation");
}

TEST(onnx2_proto, GraphProto_Nodes) {
  GraphProto graph;
  graph.set_name("test_graph");

  // Add nodes
  NodeProto& node1 = graph.add_node();
  node1.set_name("conv1");
  node1.set_op_type("Conv");
  node1.add_input() = "X";
  node1.add_input() = "W";
  node1.add_output() = "Y";

  NodeProto& node2 = graph.add_node();
  node2.set_name("relu1");
  node2.set_op_type("Relu");
  node2.add_input() = "Y";
  node2.add_output() = "Z";

  EXPECT_EQ(graph.ref_node().size(), 2);
  EXPECT_EQ(graph.ref_node()[0].ref_name(), "conv1");
  EXPECT_EQ(graph.ref_node()[0].ref_op_type(), "Conv");
  EXPECT_EQ(graph.ref_node()[1].ref_name(), "relu1");
  EXPECT_EQ(graph.ref_node()[1].ref_op_type(), "Relu");
}

TEST(onnx2_proto, GraphProto_Inputs) {
  GraphProto graph;

  ValueInfoProto& input1 = graph.add_input();
  input1.set_name("X");
  TypeProto& type1 = input1.add_type();
  type1.add_tensor_type().set_elem_type(1); // FLOAT
  TensorShapeProto& shape1 = type1.ref_tensor_type().add_shape();
  shape1.add_dim().set_dim_value(1);
  shape1.add_dim().set_dim_value(3);
  shape1.add_dim().set_dim_value(224);
  shape1.add_dim().set_dim_value(224);

  ValueInfoProto& input2 = graph.add_input();
  input2.set_name("W");
  TypeProto& type2 = input2.add_type();
  type2.add_tensor_type().set_elem_type(1); // FLOAT

  EXPECT_EQ(graph.ref_input().size(), 2);
  EXPECT_EQ(graph.ref_input()[0].ref_name(), "X");
  EXPECT_EQ(graph.ref_input()[1].ref_name(), "W");
  EXPECT_TRUE(graph.ref_input()[0].has_type());
  EXPECT_TRUE(graph.ref_input()[0].ref_type().has_tensor_type());
  EXPECT_EQ(graph.ref_input()[0].ref_type().ref_tensor_type().ref_elem_type(), 1);
}

TEST(onnx2_proto, GraphProto_Outputs) {
  GraphProto graph;

  ValueInfoProto& output = graph.add_output();
  output.set_name("Z");
  TypeProto& type = output.add_type();
  type.add_tensor_type().set_elem_type(1); // FLOAT
  TensorShapeProto& shape = type.ref_tensor_type().add_shape();
  shape.add_dim().set_dim_value(1);
  shape.add_dim().set_dim_value(64);
  shape.add_dim().set_dim_value(112);
  shape.add_dim().set_dim_value(112);

  EXPECT_EQ(graph.ref_output().size(), 1);
  EXPECT_EQ(graph.ref_output()[0].ref_name(), "Z");
  EXPECT_TRUE(graph.ref_output()[0].has_type());
  EXPECT_TRUE(graph.ref_output()[0].ref_type().has_tensor_type());
  EXPECT_TRUE(graph.ref_output()[0].ref_type().ref_tensor_type().has_shape());
  EXPECT_EQ(graph.ref_output()[0].ref_type().ref_tensor_type().ref_shape().ref_dim().size(), 4);
}

TEST(onnx2_proto, GraphProto_ValueInfo) {
  GraphProto graph;

  ValueInfoProto& value_info = graph.add_value_info();
  value_info.set_name("Y");
  TypeProto& type = value_info.add_type();
  type.add_tensor_type().set_elem_type(1); // FLOAT

  EXPECT_EQ(graph.ref_value_info().size(), 1);
  EXPECT_EQ(graph.ref_value_info()[0].ref_name(), "Y");
}

TEST(onnx2_proto, GraphProto_Initializers) {
  GraphProto graph;

  TensorProto& initializer = graph.add_initializer();
  initializer.set_name("W");
  initializer.set_data_type(TensorProto::DataType::FLOAT);
  initializer.ref_dims().push_back(64);
  initializer.ref_dims().push_back(3);
  initializer.ref_dims().push_back(3);
  initializer.ref_dims().push_back(3);

  for (int i = 0; i < 64 * 3 * 3 * 3; ++i) {
    initializer.ref_float_data().push_back(static_cast<float>(i) * 0.01f);
  }

  EXPECT_EQ(graph.ref_initializer().size(), 1);
  EXPECT_EQ(graph.ref_initializer()[0].ref_name(), "W");
  EXPECT_EQ(graph.ref_initializer()[0].ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(graph.ref_initializer()[0].ref_dims().size(), 4);
  EXPECT_EQ(graph.ref_initializer()[0].ref_float_data().size(), 64 * 3 * 3 * 3);
}

TEST(onnx2_proto, GraphProto_Serialization) {
  GraphProto graph1;
  graph1.set_name("serialization_test");

  NodeProto& node = graph1.add_node();
  node.set_name("node1");
  node.set_op_type("Identity");
  node.add_input() = "X";
  node.add_output() = "Y";

  ValueInfoProto& input = graph1.add_input();
  input.set_name("X");

  ValueInfoProto& output = graph1.add_output();
  output.set_name("Y");

  std::string serialized;
  graph1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), graph1.SerializeSize());

  GraphProto graph2;
  graph2.ParseFromString(serialized);

  EXPECT_EQ(graph2.ref_name(), "serialization_test");
  EXPECT_EQ(graph2.ref_node().size(), 1);
  EXPECT_EQ(graph2.ref_node()[0].ref_name(), "node1");
  EXPECT_EQ(graph2.ref_node()[0].ref_op_type(), "Identity");
  EXPECT_EQ(graph2.ref_input().size(), 1);
  EXPECT_EQ(graph2.ref_input()[0].ref_name(), "X");
  EXPECT_EQ(graph2.ref_output().size(), 1);
  EXPECT_EQ(graph2.ref_output()[0].ref_name(), "Y");
}

TEST(onnx2_proto, GraphProto_PrintToVectorString) {
  utils::PrintOptions options;
  GraphProto graph;
  graph.set_name("vector_serialization_test");
  graph.set_doc_string("Test graph for vector serialization");

  NodeProto& node = graph.add_node();
  node.set_name("add_node");
  node.set_op_type("Add");
  node.add_input() = "A";
  node.add_input() = "B";
  node.add_output() = "C";

  std::vector<std::string> result = graph.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  bool foundName = false;
  bool foundDocString = false;
  bool foundNode = false;

  std::string serialized = utils::join_string(result, "\n");

  if (serialized.find("name:") != std::string::npos && serialized.find("vector_serialization_test") != std::string::npos) {
    foundName = true;
  }

  if (serialized.find("doc_string:") != std::string::npos && serialized.find("Test graph for vector serialization") != std::string::npos) {
    foundDocString = true;
  }

  if (serialized.find("node") != std::string::npos && serialized.find("add_node") != std::string::npos) {
    foundNode = true;
  }

  EXPECT_TRUE(foundName);
  EXPECT_TRUE(foundDocString);
  EXPECT_TRUE(foundNode);
}

TEST(onnx2_proto, GraphProto_CopyFrom) {
  GraphProto source;
  source.set_name("source_graph");
  source.set_doc_string("Source graph documentation");

  NodeProto& node = source.add_node();
  node.set_name("test_node");
  node.set_op_type("Test");

  ValueInfoProto& input = source.add_input();
  input.set_name("input_tensor");

  ValueInfoProto& output = source.add_output();
  output.set_name("output_tensor");

  TensorProto& initializer = source.add_initializer();
  initializer.set_name("weights");
  initializer.set_data_type(TensorProto::DataType::FLOAT);

  GraphProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_name(), "source_graph");
  EXPECT_EQ(target.ref_doc_string(), "Source graph documentation");
  EXPECT_EQ(target.ref_node().size(), 1);
  EXPECT_EQ(target.ref_node()[0].ref_name(), "test_node");
  EXPECT_EQ(target.ref_input().size(), 1);
  EXPECT_EQ(target.ref_input()[0].ref_name(), "input_tensor");
  EXPECT_EQ(target.ref_output().size(), 1);
  EXPECT_EQ(target.ref_output()[0].ref_name(), "output_tensor");
  EXPECT_EQ(target.ref_initializer().size(), 1);
  EXPECT_EQ(target.ref_initializer()[0].ref_name(), "weights");
}

TEST(onnx2_proto, GraphProto_ComplexModel) {
  GraphProto graph;
  graph.set_name("complex_model");

  // Create input
  ValueInfoProto& input = graph.add_input();
  input.set_name("data");
  TypeProto& input_type = input.add_type();
  input_type.add_tensor_type().set_elem_type(1); // FLOAT
  TensorShapeProto& input_shape = input_type.ref_tensor_type().add_shape();
  input_shape.add_dim().set_dim_value(1);
  input_shape.add_dim().set_dim_value(3);
  input_shape.add_dim().set_dim_value(224);
  input_shape.add_dim().set_dim_value(224);

  // Create weights initializer
  TensorProto& weights = graph.add_initializer();
  weights.set_name("conv1_weights");
  weights.set_data_type(TensorProto::DataType::FLOAT);
  weights.ref_dims().push_back(64);
  weights.ref_dims().push_back(3);
  weights.ref_dims().push_back(7);
  weights.ref_dims().push_back(7);

  // Create bias initializer
  TensorProto& bias = graph.add_initializer();
  bias.set_name("conv1_bias");
  bias.set_data_type(TensorProto::DataType::FLOAT);
  bias.ref_dims().push_back(64);

  // Add Conv node
  NodeProto& conv = graph.add_node();
  conv.set_name("conv1");
  conv.set_op_type("Conv");
  conv.add_input() = "data";
  conv.add_input() = "conv1_weights";
  conv.add_input() = "conv1_bias";
  conv.add_output() = "conv1_output";

  AttributeProto& strides = conv.add_attribute();
  strides.set_name("strides");
  strides.set_type(AttributeProto::AttributeType::INTS);
  strides.ref_ints().push_back(2);
  strides.ref_ints().push_back(2);

  AttributeProto& kernel_shape = conv.add_attribute();
  kernel_shape.set_name("kernel_shape");
  kernel_shape.set_type(AttributeProto::AttributeType::INTS);
  kernel_shape.ref_ints().push_back(7);
  kernel_shape.ref_ints().push_back(7);

  AttributeProto& pads = conv.add_attribute();
  pads.set_name("pads");
  pads.set_type(AttributeProto::AttributeType::INTS);
  pads.ref_ints().push_back(3);
  pads.ref_ints().push_back(3);
  pads.ref_ints().push_back(3);
  pads.ref_ints().push_back(3);

  // Add ReLU node
  NodeProto& relu = graph.add_node();
  relu.set_name("relu1");
  relu.set_op_type("Relu");
  relu.add_input() = "conv1_output";
  relu.add_output() = "relu1_output";

  // Add output
  ValueInfoProto& output = graph.add_output();
  output.set_name("relu1_output");
  TypeProto& output_type = output.add_type();
  output_type.add_tensor_type().set_elem_type(1); // FLOAT

  // Add intermediate value info
  ValueInfoProto& intermediate = graph.add_value_info();
  intermediate.set_name("conv1_output");
  TypeProto& intermediate_type = intermediate.add_type();
  intermediate_type.add_tensor_type().set_elem_type(1); // FLOAT

  EXPECT_EQ(graph.ref_name(), "complex_model");
  EXPECT_EQ(graph.ref_node().size(), 2);
  EXPECT_EQ(graph.ref_initializer().size(), 2);
  EXPECT_EQ(graph.ref_input().size(), 1);
  EXPECT_EQ(graph.ref_output().size(), 1);
  EXPECT_EQ(graph.ref_value_info().size(), 1);
}

// FunctionProto

TEST(onnx2_proto, FunctionProto_Basic) {
  FunctionProto function;

  EXPECT_TRUE(function.ref_name().empty());
  EXPECT_TRUE(function.ref_domain().empty());
  EXPECT_EQ(function.ref_input().size(), 0);
  EXPECT_EQ(function.ref_output().size(), 0);
  EXPECT_EQ(function.ref_attribute().size(), 0);
  EXPECT_EQ(function.ref_node().size(), 0);
  EXPECT_TRUE(function.ref_doc_string().empty());

  function.set_name("test_function");
  function.set_domain("ai.custom");
  function.set_doc_string("Test function documentation");

  // Add inputs
  function.add_input() = "X";
  function.add_input() = "W";

  // Add outputs
  function.add_output() = "Y";

  // Add attributes
  function.add_attribute() = "alpha";
  function.add_attribute() = "beta";

  EXPECT_EQ(function.ref_name(), "test_function");
  EXPECT_EQ(function.ref_domain(), "ai.custom");
  EXPECT_EQ(function.ref_doc_string(), "Test function documentation");
  EXPECT_EQ(function.ref_input().size(), 2);
  EXPECT_EQ(function.ref_input()[0], "X");
  EXPECT_EQ(function.ref_input()[1], "W");
  EXPECT_EQ(function.ref_output().size(), 1);
  EXPECT_EQ(function.ref_output()[0], "Y");
  EXPECT_EQ(function.ref_attribute().size(), 2);
  EXPECT_EQ(function.ref_attribute()[0], "alpha");
  EXPECT_EQ(function.ref_attribute()[1], "beta");
}

TEST(onnx2_proto, FunctionProto_Nodes) {
  FunctionProto function;
  function.set_name("custom_op");

  // Add nodes
  NodeProto& node1 = function.add_node();
  node1.set_name("mul");
  node1.set_op_type("Mul");
  node1.add_input() = "X";
  node1.add_input() = "W";
  node1.add_output() = "XW";

  NodeProto& node2 = function.add_node();
  node2.set_name("add");
  node2.set_op_type("Add");
  node2.add_input() = "XW";
  node2.add_input() = "B";
  node2.add_output() = "Y";

  EXPECT_EQ(function.ref_node().size(), 2);
  EXPECT_EQ(function.ref_node()[0].ref_name(), "mul");
  EXPECT_EQ(function.ref_node()[0].ref_op_type(), "Mul");
  EXPECT_EQ(function.ref_node()[1].ref_name(), "add");
  EXPECT_EQ(function.ref_node()[1].ref_op_type(), "Add");
}

TEST(onnx2_proto, FunctionProto_Serialization) {
  FunctionProto function1;
  function1.set_name("serialization_function");
  function1.set_domain("ai.test");
  function1.set_doc_string("Function for serialization testing");

  function1.add_input() = "X";
  function1.add_output() = "Y";
  function1.add_attribute() = "param";

  NodeProto& node = function1.add_node();
  node.set_name("op1");
  node.set_op_type("CustomOp");
  node.add_input() = "X";
  node.add_output() = "Y";

  std::string serialized;
  function1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), function1.SerializeSize());

  FunctionProto function2;
  function2.ParseFromString(serialized);

  EXPECT_EQ(function2.ref_name(), "serialization_function");
  EXPECT_EQ(function2.ref_domain(), "ai.test");
  EXPECT_EQ(function2.ref_doc_string(), "Function for serialization testing");
  EXPECT_EQ(function2.ref_input().size(), 1);
  EXPECT_EQ(function2.ref_input()[0], "X");
  EXPECT_EQ(function2.ref_output().size(), 1);
  EXPECT_EQ(function2.ref_output()[0], "Y");
  EXPECT_EQ(function2.ref_attribute().size(), 1);
  EXPECT_EQ(function2.ref_attribute()[0], "param");
  EXPECT_EQ(function2.ref_node().size(), 1);
  EXPECT_EQ(function2.ref_node()[0].ref_name(), "op1");
}

TEST(onnx2_proto, FunctionProto_CopyFrom) {
  FunctionProto source;
  source.set_name("source_function");
  source.set_domain("ai.source");
  source.add_input() = "X";
  source.add_output() = "Y";
  source.add_attribute() = "attr1";

  NodeProto& node = source.add_node();
  node.set_op_type("Identity");

  source.set_doc_string("Source function documentation");

  FunctionProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_name(), "source_function");
  EXPECT_EQ(target.ref_domain(), "ai.source");
  EXPECT_EQ(target.ref_input().size(), 1);
  EXPECT_EQ(target.ref_input()[0], "X");
  EXPECT_EQ(target.ref_output().size(), 1);
  EXPECT_EQ(target.ref_output()[0], "Y");
  EXPECT_EQ(target.ref_attribute().size(), 1);
  EXPECT_EQ(target.ref_attribute()[0], "attr1");
  EXPECT_EQ(target.ref_node().size(), 1);
  EXPECT_EQ(target.ref_node()[0].ref_op_type(), "Identity");
  EXPECT_EQ(target.ref_doc_string(), "Source function documentation");
}

TEST(onnx2_string, FunctionProto) {
  utils::PrintOptions options;
  FunctionProto function;
  function.set_name("my_function");
  function.set_domain("ai.custom");
  function.add_input() = "input1";
  function.add_input() = "input2";
  function.add_output() = "output";
  function.add_attribute() = "attr";
  function.set_doc_string("Custom function implementation");

  NodeProto& node = function.add_node();
  node.set_name("operation");
  node.set_op_type("MatMul");

  std::vector<std::string> result = function.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  std::string serialized = utils::join_string(result, "\n");

  EXPECT_TRUE(serialized.find("name: \"my_function\"") != std::string::npos);
  EXPECT_TRUE(serialized.find("domain: \"ai.custom\"") != std::string::npos);
  EXPECT_TRUE(serialized.find("input:") != std::string::npos);
  EXPECT_TRUE(serialized.find("output:") != std::string::npos);
  EXPECT_TRUE(serialized.find("attribute:") != std::string::npos);
  EXPECT_TRUE(serialized.find("node: [") != std::string::npos);
  EXPECT_TRUE(serialized.find("doc_string:") != std::string::npos);
}

// ModelProto

TEST(onnx2_proto, ModelProto_Basic) {
  ModelProto model;

  EXPECT_TRUE(model.ref_producer_name().empty());
  EXPECT_TRUE(model.ref_producer_version().empty());
  EXPECT_TRUE(model.ref_domain().empty());
  EXPECT_EQ(model.ref_model_version(), 0);
  EXPECT_TRUE(model.ref_doc_string().empty());
  EXPECT_FALSE(model.has_graph());
  EXPECT_EQ(model.ref_opset_import().size(), 0);
  EXPECT_EQ(model.ref_metadata_props().size(), 0);

  model.set_ir_version(1);
  model.set_producer_name("test_producer");
  model.set_producer_version("1.0.0");
  model.set_domain("ai.test");
  model.set_model_version(1);
  model.set_doc_string("Test model documentation");

  EXPECT_EQ(model.ref_ir_version(), 1);
  EXPECT_EQ(model.ref_producer_name(), "test_producer");
  EXPECT_EQ(model.ref_producer_version(), "1.0.0");
  EXPECT_EQ(model.ref_domain(), "ai.test");
  EXPECT_EQ(model.ref_model_version(), 1);
  EXPECT_EQ(model.ref_doc_string(), "Test model documentation");
}

TEST(onnx2_proto, ModelProto_Graph) {
  ModelProto model;

  GraphProto& graph = model.add_graph();
  graph.set_name("test_graph");

  NodeProto& node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Add");

  EXPECT_TRUE(model.has_graph());
  EXPECT_EQ(model.ref_graph().ref_name(), "test_graph");
  EXPECT_EQ(model.ref_graph().ref_node().size(), 1);
  EXPECT_EQ(model.ref_graph().ref_node()[0].ref_name(), "test_node");
  EXPECT_EQ(model.ref_graph().ref_node()[0].ref_op_type(), "Add");
}

TEST(onnx2_proto, ModelProto_OpsetImport) {
  ModelProto model;

  OperatorSetIdProto& opset1 = model.add_opset_import();
  opset1.set_domain("ai.onnx");
  opset1.set_version(12);

  OperatorSetIdProto& opset2 = model.add_opset_import();
  opset2.set_domain("ai.onnx.ml");
  opset2.set_version(2);

  EXPECT_EQ(model.ref_opset_import().size(), 2);
  EXPECT_EQ(model.ref_opset_import()[0].ref_domain(), "ai.onnx");
  EXPECT_EQ(model.ref_opset_import()[0].ref_version(), 12);
  EXPECT_EQ(model.ref_opset_import()[1].ref_domain(), "ai.onnx.ml");
  EXPECT_EQ(model.ref_opset_import()[1].ref_version(), 2);
}

TEST(onnx2_proto, ModelProto_MetadataProps) {
  ModelProto model;

  StringStringEntryProto& metadata1 = model.add_metadata_props();
  metadata1.set_key("author");
  metadata1.set_value("test_author");

  StringStringEntryProto& metadata2 = model.add_metadata_props();
  metadata2.set_key("description");
  metadata2.set_value("test description");

  EXPECT_EQ(model.ref_metadata_props().size(), 2);
  EXPECT_EQ(model.ref_metadata_props()[0].ref_key(), "author");
  EXPECT_EQ(model.ref_metadata_props()[0].ref_value(), "test_author");
  EXPECT_EQ(model.ref_metadata_props()[1].ref_key(), "description");
  EXPECT_EQ(model.ref_metadata_props()[1].ref_value(), "test description");
}

TEST(onnx2_proto, ModelProto_Serialization) {
  ModelProto model1;
  model1.set_ir_version(1);
  model1.set_producer_name("serialization_test");
  model1.set_model_version(42);

  GraphProto& graph = model1.add_graph();
  graph.set_name("serialized_graph");

  NodeProto& node = graph.add_node();
  node.set_name("test_node");
  node.set_op_type("Identity");

  StringStringEntryProto& metadata = model1.add_metadata_props();
  metadata.set_key("test_key");
  metadata.set_value("test_value");

  std::string serialized;
  model1.SerializeToString(serialized);
  EXPECT_FALSE(serialized.empty());
  EXPECT_EQ(serialized.size(), model1.SerializeSize());

  ModelProto model2;
  model2.ParseFromString(serialized);

  EXPECT_EQ(model2.ref_ir_version(), 1);
  EXPECT_EQ(model2.ref_producer_name(), "serialization_test");
  EXPECT_EQ(model2.ref_model_version(), 42);
  EXPECT_TRUE(model2.has_graph());
  EXPECT_EQ(model2.ref_graph().ref_name(), "serialized_graph");
  EXPECT_EQ(model2.ref_graph().ref_node().size(), 1);
  EXPECT_EQ(model2.ref_graph().ref_node()[0].ref_name(), "test_node");
  EXPECT_EQ(model2.ref_metadata_props().size(), 1);
  EXPECT_EQ(model2.ref_metadata_props()[0].ref_key(), "test_key");
  EXPECT_EQ(model2.ref_metadata_props()[0].ref_value(), "test_value");
}

TEST(onnx2_proto, ModelProto_PrintToVectorString) {
  utils::PrintOptions options;
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("test_producer");
  model.set_doc_string("Model documentation");
  model.add_graph().set_name("test_graph");

  std::vector<std::string> result = model.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  std::string serialized = utils::join_string(result, "\n");

  EXPECT_TRUE(serialized.find("ir_version:") != std::string::npos);
  EXPECT_TRUE(serialized.find("producer_name:") != std::string::npos);
  EXPECT_TRUE(serialized.find("test_producer") != std::string::npos);
  EXPECT_TRUE(serialized.find("doc_string:") != std::string::npos);
  EXPECT_TRUE(serialized.find("Model documentation") != std::string::npos);
  EXPECT_TRUE(serialized.find("graph") != std::string::npos);
  EXPECT_TRUE(serialized.find("test_graph") != std::string::npos);
}

TEST(onnx2_proto, ModelProto_CopyFrom) {
  ModelProto source;
  source.set_ir_version(2);
  source.set_producer_name("source_producer");
  source.set_model_version(123);
  source.add_graph().set_name("source_graph");

  OperatorSetIdProto& opset = source.add_opset_import();
  opset.set_domain("ai.onnx");
  opset.set_version(15);

  StringStringEntryProto& metadata = source.add_metadata_props();
  metadata.set_key("source_key");
  metadata.set_value("source_value");

  ModelProto target;
  target.CopyFrom(source);

  EXPECT_EQ(target.ref_ir_version(), 2);
  EXPECT_EQ(target.ref_producer_name(), "source_producer");
  EXPECT_EQ(target.ref_model_version(), 123);
  EXPECT_TRUE(target.has_graph());
  EXPECT_EQ(target.ref_graph().ref_name(), "source_graph");
  EXPECT_EQ(target.ref_opset_import().size(), 1);
  EXPECT_EQ(target.ref_opset_import()[0].ref_domain(), "ai.onnx");
  EXPECT_EQ(target.ref_opset_import()[0].ref_version(), 15);
  EXPECT_EQ(target.ref_metadata_props().size(), 1);
  EXPECT_EQ(target.ref_metadata_props()[0].ref_key(), "source_key");
  EXPECT_EQ(target.ref_metadata_props()[0].ref_value(), "source_value");
}

TEST(onnx2_proto, ModelProto_ComplexModel) {
  ModelProto model;
  model.set_ir_version(3);
  model.set_producer_name("complex_model_producer");
  model.set_producer_version("1.0.0");
  model.set_model_version(1);

  OperatorSetIdProto& opset = model.add_opset_import();
  opset.set_domain("ai.onnx");
  opset.set_version(13);

  GraphProto& graph = model.add_graph();
  graph.set_name("complex_model_graph");

  // Add input
  ValueInfoProto& input = graph.add_input();
  input.set_name("input_tensor");
  TypeProto& input_type = input.add_type();
  input_type.add_tensor_type().set_elem_type(1); // FLOAT

  // Add initializer
  TensorProto& weights = graph.add_initializer();
  weights.set_name("weights");
  weights.set_data_type(TensorProto::DataType::FLOAT);
  weights.ref_dims().push_back(3);
  weights.ref_dims().push_back(3);

  // Add node
  NodeProto& node = graph.add_node();
  node.set_name("matmul_node");
  node.set_op_type("MatMul");
  node.add_input() = "input_tensor";
  node.add_input() = "weights";
  node.add_output() = "output_tensor";

  // Add output
  ValueInfoProto& output = graph.add_output();
  output.set_name("output_tensor");

  // Add metadata
  StringStringEntryProto& metadata = model.add_metadata_props();
  metadata.set_key("framework");
  metadata.set_value("test_framework");

  EXPECT_EQ(model.ref_ir_version(), 3);
  EXPECT_EQ(model.ref_producer_name(), "complex_model_producer");
  EXPECT_EQ(model.ref_model_version(), 1);
  EXPECT_TRUE(model.has_graph());

  EXPECT_EQ(model.ref_graph().ref_input().size(), 1);
  EXPECT_EQ(model.ref_graph().ref_initializer().size(), 1);
  EXPECT_EQ(model.ref_graph().ref_node().size(), 1);
  EXPECT_EQ(model.ref_graph().ref_output().size(), 1);

  EXPECT_EQ(model.ref_opset_import().size(), 1);
  EXPECT_EQ(model.ref_opset_import()[0].ref_version(), 13);

  EXPECT_EQ(model.ref_metadata_props().size(), 1);
  EXPECT_EQ(model.ref_metadata_props()[0].ref_key(), "framework");
}

TEST(onnx2_proto, AttributeProto_InNodeProto1) {
  utils::PrintOptions options;
  NodeProto node;
  node.set_name("test_node");
  node.set_op_type("TestOp");
  AttributeProto& attr1 = node.add_attribute();
  attr1.set_type(AttributeProto::AttributeType::INT);
  attr1.ref_i() = 2;
  AttributeProto att2;
  att2.set_type(AttributeProto::AttributeType::INT);
  att2.ref_i() = 2;
  node.ref_attribute().push_back(att2);
  std::string s1 = node.ref_attribute()[0].PrintToVectorString(options)[0];
  std::string s2 = node.ref_attribute()[1].PrintToVectorString(options)[0];
  EXPECT_EQ(s1, s2);
  std::string s4 = att2.PrintToVectorString(options)[0];
  EXPECT_EQ(s1, s4);
}

TEST(onnx2_proto, AttributeProto_InNodeProto2) {
  utils::PrintOptions options;
  NodeProto node;
  node.set_name("test_node");
  node.set_op_type("TestOp");
  AttributeProto& attr1 = node.add_attribute();
  attr1.set_type(AttributeProto::AttributeType::INT);
  attr1.ref_i() = 2;
  AttributeProto& att2 = node.add_attribute();
  att2.set_type(AttributeProto::AttributeType::INT);
  att2.ref_i() = 2;
  std::string s1 = node.ref_attribute()[0].PrintToVectorString(options)[0];
  std::string s2 = node.ref_attribute()[1].PrintToVectorString(options)[0];
  EXPECT_EQ(s1, s2);
  std::string s4 = att2.PrintToVectorString(options)[0];
  EXPECT_EQ(s1, s4);
}

TEST(onnx2_proto, TensorProto_SkipRawData) {
  TensorProto tensor1;
  tensor1.set_name("skip_raw_test");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.ref_dims().push_back(2);
  tensor1.ref_dims().push_back(2);

  // Ajout de donn�es brutes
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  tensor1.ref_raw_data().resize(data.size() * sizeof(float));
  std::memcpy(tensor1.ref_raw_data().data(), data.data(), data.size() * sizeof(float));

  std::string serialized1;
  SerializeOptions options;
  utils::StringWriteStream st;
  tensor1.SerializeToString(serialized1, options);
  EXPECT_EQ(serialized1.size(), tensor1.SerializeSize(st, options));

  std::string serialized2;
  SerializeOptions options2;
  options2.skip_raw_data = true;
  options2.raw_data_threshold = 0;
  tensor1.SerializeToString(serialized2, options2);
  EXPECT_EQ(serialized1.size(), 39);
  EXPECT_EQ(serialized2.size(), 21);
  EXPECT_EQ(serialized2.size(), tensor1.SerializeSize(st, options2));

  // Test avec skip_raw_data = false (comportement par d�faut)
  ParseOptions parse_options;
  parse_options.skip_raw_data = true;
  parse_options.raw_data_threshold = 0;
  TensorProto tensor2;
  tensor2.ParseFromString(serialized1, parse_options);

  EXPECT_EQ(tensor2.ref_name(), "skip_raw_test");
  EXPECT_EQ(tensor2.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor2.ref_dims().size(), 2);
  EXPECT_EQ(tensor2.ref_raw_data().size(), 0);
}

TEST(onnx2_stream, FileWriteStream) {
  std::string temp_filename = "test_file_write_stream.tmp";

  // read
  {
    utils::FileWriteStream stream(temp_filename);

    stream.write_variant_uint64(150);
    stream.write_int64(42);
    stream.write_int32(24);
    stream.write_float(3.14f);
    stream.write_string("hello");

    EXPECT_GT(stream.size(), 0);
  }

  // check content
  {
    utils::FileStream readStream(temp_filename);

    EXPECT_EQ(readStream.next_uint64(), 150);
    EXPECT_EQ(readStream.next_int64(), 42);
    EXPECT_EQ(readStream.next_int32(), 24);
    EXPECT_NEAR(readStream.next_float(), 3.14f, 0.0001f);

    utils::RefString str = readStream.next_string();
    EXPECT_EQ(str, "hello");
    EXPECT_FALSE(readStream.NotEnd());
  }

  // Clean up
  std::remove(temp_filename.c_str());
}

TEST(onnx2_stream, FileStream_TensorProto) {
  std::string temp_filename = "test_tensor_file_stream.tmp";

  // create and save a TensorProto to a file
  {
    TensorProto tensor;
    tensor.set_name("test_tensor");
    tensor.set_data_type(TensorProto::DataType::FLOAT);
    tensor.ref_dims().push_back(2);
    tensor.ref_dims().push_back(3);

    // Add float data
    tensor.ref_float_data().push_back(1.1f);
    tensor.ref_float_data().push_back(2.2f);
    tensor.ref_float_data().push_back(3.3f);
    tensor.ref_float_data().push_back(4.4f);
    tensor.ref_float_data().push_back(5.5f);
    tensor.ref_float_data().push_back(6.6f);

    // Serialize to a file
    utils::FileWriteStream stream(temp_filename);
    std::string serialized;
    tensor.SerializeToString(serialized);

    // Write the size followed by the serialized data
    stream.write_variant_uint64(serialized.size());
    stream.write_raw_bytes(reinterpret_cast<const uint8_t*>(serialized.data()), serialized.size());
  }

  // Read and deserialize the TensorProto from the file
  {
    utils::FileStream stream(temp_filename);

    // Read the size and data
    uint64_t size = stream.next_uint64();
    std::vector<uint8_t> buffer(size);
    const uint8_t* data = stream.read_bytes(size);
    std::memcpy(buffer.data(), data, size);

    // Deserialize the TensorProto
    TensorProto tensor;
    tensor.ParseFromString(std::string(reinterpret_cast<const char *>(buffer.data()), buffer.size()));

    // Check properties
    EXPECT_EQ(tensor.ref_name(), "test_tensor");
    EXPECT_EQ(tensor.ref_data_type(), TensorProto::DataType::FLOAT);
    EXPECT_EQ(tensor.ref_dims().size(), 2);
    EXPECT_EQ(tensor.ref_dims()[0], 2);
    EXPECT_EQ(tensor.ref_dims()[1], 3);

    // Check data
    ASSERT_EQ(tensor.ref_float_data().size(), 6);
    EXPECT_FLOAT_EQ(tensor.ref_float_data()[0], 1.1f);
    EXPECT_FLOAT_EQ(tensor.ref_float_data()[1], 2.2f);
    EXPECT_FLOAT_EQ(tensor.ref_float_data()[2], 3.3f);
    EXPECT_FLOAT_EQ(tensor.ref_float_data()[3], 4.4f);
    EXPECT_FLOAT_EQ(tensor.ref_float_data()[4], 5.5f);
    EXPECT_FLOAT_EQ(tensor.ref_float_data()[5], 6.6f);
  }

  // Clean up
  std::remove(temp_filename.c_str());
}

TEST(onnx2_proto, AttributeProto_TensorsAttribute) {
  AttributeProto attribute;

  attribute.set_name("weights");
  attribute.set_type(AttributeProto::AttributeType::TENSORS);

  TensorProto& tensor1 = attribute.add_tensors();
  tensor1.set_name("tensor1");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.ref_dims().push_back(2);
  tensor1.ref_dims().push_back(3);
  tensor1.ref_float_data().push_back(1.0f);
  tensor1.ref_float_data().push_back(2.0f);
  tensor1.ref_float_data().push_back(3.0f);
  tensor1.ref_float_data().push_back(4.0f);
  tensor1.ref_float_data().push_back(5.0f);
  tensor1.ref_float_data().push_back(6.0f);

  TensorProto& tensor2 = attribute.add_tensors();
  tensor2.set_name("tensor2");
  tensor2.set_data_type(TensorProto::DataType::INT32);
  tensor2.ref_dims().push_back(2);
  tensor2.ref_int32_data().push_back(10);
  tensor2.ref_int32_data().push_back(20);

  EXPECT_EQ(attribute.ref_name(), "weights");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::TENSORS);
  EXPECT_EQ(attribute.ref_tensors().size(), 2);
  EXPECT_EQ(attribute.ref_tensors()[0].ref_name(), "tensor1");
  EXPECT_EQ(attribute.ref_tensors()[0].ref_float_data().size(), 6);
  EXPECT_EQ(attribute.ref_tensors()[1].ref_name(), "tensor2");
  EXPECT_EQ(attribute.ref_tensors()[1].ref_int32_data().size(), 2);
}

TEST(onnx2_proto, AttributeProto_GraphsAttribute) {
  AttributeProto attribute;

  attribute.set_name("branches");
  attribute.set_type(AttributeProto::AttributeType::GRAPHS);

  GraphProto& graph1 = attribute.add_graphs();
  graph1.set_name("if_branch");
  NodeProto& node1 = graph1.add_node();
  node1.set_name("add_node");
  node1.set_op_type("Add");

  GraphProto& graph2 = attribute.add_graphs();
  graph2.set_name("else_branch");
  NodeProto& node2 = graph2.add_node();
  node2.set_name("mul_node");
  node2.set_op_type("Mul");

  EXPECT_EQ(attribute.ref_name(), "branches");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::GRAPHS);
  EXPECT_EQ(attribute.ref_graphs().size(), 2);
  EXPECT_EQ(attribute.ref_graphs()[0].ref_name(), "if_branch");
  EXPECT_EQ(attribute.ref_graphs()[0].ref_node()[0].ref_op_type(), "Add");
  EXPECT_EQ(attribute.ref_graphs()[1].ref_name(), "else_branch");
  EXPECT_EQ(attribute.ref_graphs()[1].ref_node()[0].ref_op_type(), "Mul");
}

TEST(onnx2_proto, AttributeProto_DocString) {
  AttributeProto attribute;
  attribute.set_name("dropout_ratio");
  attribute.set_type(AttributeProto::AttributeType::FLOAT);
  attribute.set_f(0.5f);
  attribute.set_doc_string("Controls the rate at which activations are dropped");

  EXPECT_EQ(attribute.ref_name(), "dropout_ratio");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::FLOAT);
  EXPECT_EQ(attribute.ref_f(), 0.5f);
  EXPECT_EQ(attribute.ref_doc_string(), "Controls the rate at which activations are dropped");
}

TEST(onnx2_proto, AttributeProto_Serialization_AllTypes_INT) {
  {
    // Test INT attribute
    AttributeProto int_attr;
    int_attr.set_name("int_attr");
    int_attr.set_type(AttributeProto::AttributeType::INT);
    int_attr.set_i(42);

    std::string serialized;
    int_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "int_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::INT);
    EXPECT_EQ(deserialized.ref_i(), 42);
  }
}

TEST(onnx2_proto, AttributeProto_Serialization_AllTypes_FLOAT) {
  {
    // Test FLOAT attribute
    AttributeProto float_attr;
    float_attr.set_name("float_attr");
    float_attr.set_type(AttributeProto::AttributeType::FLOAT);
    float_attr.set_f(3.14f);

    std::string serialized;
    float_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "float_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::FLOAT);
    EXPECT_FLOAT_EQ(deserialized.ref_f(), 3.14f);
  }
}

TEST(onnx2_proto, AttributeProto_Serialization_AllTypes_STRING) {
  {
    // Test STRING attribute
    AttributeProto string_attr;
    string_attr.set_name("string_attr");
    string_attr.set_type(AttributeProto::AttributeType::STRING);
    string_attr.set_s("test_string");

    std::string serialized;
    string_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "string_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::STRING);
    EXPECT_EQ(deserialized.ref_s(), "test_string");
  }
}

TEST(onnx2_proto, AttributeProto_Serialization_AllTypes_INTS) {
  {
    // Test INTS attribute
    AttributeProto ints_attr;
    ints_attr.set_name("ints_attr");
    ints_attr.set_type(AttributeProto::AttributeType::INTS);
    ints_attr.ref_ints().push_back(1);
    ints_attr.ref_ints().push_back(2);
    ints_attr.ref_ints().push_back(3);

    std::string serialized;
    ints_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "ints_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::INTS);
    EXPECT_EQ(deserialized.ref_ints().size(), 3);
    EXPECT_EQ(deserialized.ref_ints()[0], 1);
    EXPECT_EQ(deserialized.ref_ints()[1], 2);
    EXPECT_EQ(deserialized.ref_ints()[2], 3);
  }
}

TEST(onnx2_proto, AttributeProto_Serialization_AllTypes_FLOATS) {
  {
    // Test FLOATS attribute
    AttributeProto floats_attr;
    floats_attr.set_name("floats_attr");
    floats_attr.set_type(AttributeProto::AttributeType::FLOATS);
    floats_attr.ref_floats().push_back(1.1f);
    floats_attr.ref_floats().push_back(2.2f);

    std::string serialized;
    floats_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "floats_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::FLOATS);
    EXPECT_EQ(deserialized.ref_floats().size(), 2);
    EXPECT_FLOAT_EQ(deserialized.ref_floats()[0], 1.1f);
    EXPECT_FLOAT_EQ(deserialized.ref_floats()[1], 2.2f);
  }
}

TEST(onnx2_proto, AttributeProto_Serialization_AllTypes_TENSOR) {
  {
    // Test TENSOR attribute
    AttributeProto tensor_attr;
    tensor_attr.set_name("tensor_attr");
    tensor_attr.set_type(AttributeProto::AttributeType::TENSOR);
    tensor_attr.ref_t().set_data_type(TensorProto::DataType::FLOAT);
    tensor_attr.ref_t().add_dims() = 2;
    tensor_attr.ref_t().add_dims() = 3;
    tensor_attr.ref_t().ref_float_data().add() = 1.1f;
    tensor_attr.ref_t().ref_float_data().add() = 2.2f;

    std::string serialized;
    tensor_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "tensor_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::TENSOR);
    EXPECT_EQ(deserialized.ref_t().ref_data_type(), TensorProto::DataType::FLOAT);
    EXPECT_EQ(deserialized.ref_t().ref_dims().size(), 2);
    EXPECT_EQ(deserialized.ref_t().ref_dims()[0], 2);
    EXPECT_EQ(deserialized.ref_t().ref_dims()[1], 3);
    EXPECT_EQ(deserialized.ref_t().ref_float_data().size(), 2);
    EXPECT_FLOAT_EQ(deserialized.ref_t().ref_float_data()[0], 1.1f);
    EXPECT_FLOAT_EQ(deserialized.ref_t().ref_float_data()[1], 2.2f);
  }
}

TEST(onnx2_proto, AttributeProto_Serialization_AllTypes_STRINGS) {
  {
    // Test STRINGS attribute
    AttributeProto strings_attr;
    strings_attr.set_name("strings_attr");
    strings_attr.set_type(AttributeProto::AttributeType::STRINGS);
    strings_attr.ref_strings().push_back(utils::String("test_string_1"));
    strings_attr.ref_strings().push_back(utils::String("test_string_2"));

    std::string serialized;
    strings_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "strings_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::STRINGS);
    EXPECT_EQ(deserialized.ref_strings().size(), 2);
    EXPECT_EQ(deserialized.ref_strings()[0], "test_string_1");
    EXPECT_EQ(deserialized.ref_strings()[1], "test_string_2");
  }
}

TEST(onnx2_proto, AttributeProto_Serialization_AllTypes_GRAPH) {
  {
    // Test GRAPH attribute
    AttributeProto graph_attr;
    graph_attr.set_name("graph_attr");
    graph_attr.set_type(AttributeProto::AttributeType::GRAPH);
    graph_attr.ref_g().set_name("test_graph");

    std::string serialized;
    graph_attr.SerializeToString(serialized);

    AttributeProto deserialized;
    deserialized.ParseFromString(serialized);

    EXPECT_EQ(deserialized.ref_name(), "graph_attr");
    EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::GRAPH);
    EXPECT_EQ(deserialized.ref_g().ref_name(), "test_graph");
  }
}

TEST(onnx2_proto, AttributeProto_PrintToVectorString_AllTypes) {
  utils::PrintOptions options;

  {
    // Test INT attribute print
    AttributeProto int_attr;
    int_attr.set_name("int_attr");
    int_attr.set_type(AttributeProto::AttributeType::INT);
    int_attr.set_i(42);

    std::vector<std::string> result = int_attr.PrintToVectorString(options);
    std::string serialized = utils::join_string(result, "\n");

    EXPECT_TRUE(serialized.find("int_attr: 42") != std::string::npos);
  }

  {
    // Test INTS attribute print
    AttributeProto ints_attr;
    ints_attr.set_name("ints_attr");
    ints_attr.set_type(AttributeProto::AttributeType::INTS);
    ints_attr.ref_ints().push_back(1);
    ints_attr.ref_ints().push_back(2);
    ints_attr.ref_ints().push_back(3);

    std::vector<std::string> result = ints_attr.PrintToVectorString(options);
    std::string serialized = utils::join_string(result, "\n");

    EXPECT_TRUE(serialized.find("ints_attr: [1, 2, 3]") != std::string::npos);
  }

  {
    // Test FLOATS attribute print
    AttributeProto floats_attr;
    floats_attr.set_name("floats_attr");
    floats_attr.set_type(AttributeProto::AttributeType::FLOATS);
    floats_attr.ref_floats().push_back(1.1f);
    floats_attr.ref_floats().push_back(2.2f);

    std::vector<std::string> result = floats_attr.PrintToVectorString(options);
    std::string serialized = utils::join_string(result, "\n");

    EXPECT_TRUE(serialized.find("floats_attr: [1.1, 2.2]") != std::string::npos);
  }
}

TEST(onnx2_proto, AttributeProto_EmptyCollectionAttributes) {
  // Test empty INTS
  AttributeProto ints_attr;
  ints_attr.set_name("empty_ints");
  ints_attr.set_type(AttributeProto::AttributeType::INTS);

  EXPECT_EQ(ints_attr.ref_name(), "empty_ints");
  EXPECT_EQ(ints_attr.ref_type(), AttributeProto::AttributeType::INTS);
  EXPECT_EQ(ints_attr.ref_ints().size(), 0);

  // Test empty FLOATS
  AttributeProto floats_attr;
  floats_attr.set_name("empty_floats");
  floats_attr.set_type(AttributeProto::AttributeType::FLOATS);

  EXPECT_EQ(floats_attr.ref_name(), "empty_floats");
  EXPECT_EQ(floats_attr.ref_type(), AttributeProto::AttributeType::FLOATS);
  EXPECT_EQ(floats_attr.ref_floats().size(), 0);

  // Test empty STRINGS
  AttributeProto strings_attr;
  strings_attr.set_name("empty_strings");
  strings_attr.set_type(AttributeProto::AttributeType::STRINGS);

  EXPECT_EQ(strings_attr.ref_name(), "empty_strings");
  EXPECT_EQ(strings_attr.ref_type(), AttributeProto::AttributeType::STRINGS);
  EXPECT_EQ(strings_attr.ref_strings().size(), 0);

  // Test empty TENSORS
  AttributeProto tensors_attr;
  tensors_attr.set_name("empty_tensors");
  tensors_attr.set_type(AttributeProto::AttributeType::TENSORS);

  EXPECT_EQ(tensors_attr.ref_name(), "empty_tensors");
  EXPECT_EQ(tensors_attr.ref_type(), AttributeProto::AttributeType::TENSORS);
  EXPECT_EQ(tensors_attr.ref_tensors().size(), 0);
}

TEST(onnx2_proto, AttributeProto_RefVersusAccessors) {
  AttributeProto attr;
  attr.set_name("test_attr");

  // Test INT
  attr.set_type(AttributeProto::AttributeType::INT);
  attr.set_i(42);
  EXPECT_EQ(attr.ref_i(), 42);
  EXPECT_TRUE(attr.has_i());

  // Test FLOAT
  attr.set_type(AttributeProto::AttributeType::FLOAT);
  attr.set_f(3.14f);
  EXPECT_FLOAT_EQ(attr.ref_f(), 3.14f);
  EXPECT_TRUE(attr.has_f());

  // Test STRING
  attr.set_type(AttributeProto::AttributeType::STRING);
  attr.set_s("test_string");
  EXPECT_EQ(attr.ref_s(), "test_string");
  EXPECT_TRUE(attr.has_s());

  // Test TENSOR
  attr.set_type(AttributeProto::AttributeType::TENSOR);
  TensorProto& tensor = attr.add_t();
  tensor.set_name("tensor_name");
  EXPECT_EQ(attr.ref_t().ref_name(), "tensor_name");
  EXPECT_TRUE(attr.has_t());

  // Test GRAPH
  attr.set_type(AttributeProto::AttributeType::GRAPH);
  GraphProto& graph = attr.add_g();
  graph.set_name("graph_name");
  EXPECT_EQ(attr.ref_g().ref_name(), "graph_name");
  EXPECT_TRUE(attr.has_g());
}

// check size of AttributeProto serialization with the function returning the size

TEST(onnx2_proto, SerializeSize_AttributeProto) {
  AttributeProto attribute;
  attribute.set_name("test_attribute");
  attribute.set_type(AttributeProto::AttributeType::INT);
  attribute.set_i(42);
  attribute.set_doc_string("Test attribute documentation");

  std::string serialized;
  attribute.SerializeToString(serialized);
  utils::StringWriteStream stream;
  SerializeOptions options;
  EXPECT_EQ(serialized.size(), attribute.SerializeSize(stream, options));
}

TEST(onnx2_proto, SerializeSize_AttributeProto_EmptyStrings) {
  AttributeProto attribute;
  attribute.set_name("");
  attribute.set_type(AttributeProto::AttributeType::STRING);
  attribute.set_s("");
  attribute.set_doc_string("");

  std::string serialized;
  attribute.SerializeToString(serialized);
  utils::StringWriteStream stream;
  SerializeOptions options;
  EXPECT_EQ(serialized.size(), attribute.SerializeSize(stream, options));
}

TEST(onnx2_proto, SerializeSize_AttributeProto_NullStrings) {
  AttributeProto attribute;
  // Do not set name, s, or doc_string to simulate null strings
  attribute.set_type(AttributeProto::AttributeType::STRING);

  std::string serialized;
  attribute.SerializeToString(serialized);
  utils::StringWriteStream stream;
  SerializeOptions options;
  EXPECT_EQ(serialized.size(), attribute.SerializeSize(stream, options));
}

TEST(onnx2_proto, SerializeSize_String) {
  utils::String test_string("hello world", 11);

  std::string serialized;
  utils::StringWriteStream write_stream;
  write_stream.write_string(test_string);

  utils::StringStream read_stream(write_stream.data(), write_stream.size());
  utils::RefString read_string = read_stream.next_string();

  EXPECT_EQ(write_stream.size(), test_string.size() + write_stream.size_variant_uint64(test_string.size()));
  EXPECT_EQ(read_string, test_string);
}

TEST(onnx2_proto, SerializeSize_EmptyString) {
  utils::String empty_string("", 0);

  utils::StringWriteStream write_stream;
  write_stream.write_string(empty_string);

  utils::StringStream read_stream(write_stream.data(), write_stream.size());
  utils::RefString read_string = read_stream.next_string();

  EXPECT_EQ(write_stream.size(), write_stream.size_variant_uint64(0));
  EXPECT_EQ(read_string.size(), 0);
  EXPECT_TRUE(read_string.empty());
}

TEST(onnx2_proto, SerializeSize_NullString) {
  utils::String null_string;

  utils::StringWriteStream write_stream;
  write_stream.write_string(null_string);

  utils::StringStream read_stream(write_stream.data(), write_stream.size());
  utils::RefString read_string = read_stream.next_string();

  EXPECT_EQ(write_stream.size(), write_stream.size_variant_uint64(0));
  EXPECT_EQ(read_string.size(), 0);
  EXPECT_TRUE(read_string.empty());
}

TEST(onnx2_proto, SerializeSize_StringWithNulls) {
  std::vector<char> data = {'t', 'e', 's', 't', '\0', 'n', 'u', 'l', 'l'};
  utils::String string_with_nulls(data.data(), data.size());

  utils::StringWriteStream write_stream;
  write_stream.write_string(string_with_nulls);

  utils::StringStream read_stream(write_stream.data(), write_stream.size());
  utils::RefString read_string = read_stream.next_string();

  EXPECT_EQ(write_stream.size(), string_with_nulls.size() + write_stream.size_variant_uint64(string_with_nulls.size()));
  EXPECT_EQ(read_string.size(), string_with_nulls.size());
}

TEST(onnx2_proto, SerializeSize_AttributeProto_IntFloatTensors) {
  AttributeProto attribute;
  attribute.set_name("complex_attribute");
  attribute.set_type(AttributeProto::AttributeType::TENSORS);

  TensorProto& tensor1 = attribute.add_tensors();
  tensor1.set_name("tensor1");
  tensor1.set_data_type(TensorProto::DataType::FLOAT);
  tensor1.ref_dims().push_back(2);
  tensor1.ref_dims().push_back(3);
  tensor1.ref_float_data().push_back(1.0f);
  tensor1.ref_float_data().push_back(2.0f);

  TensorProto& tensor2 = attribute.add_tensors();
  tensor2.set_name("tensor2");
  tensor2.set_data_type(TensorProto::DataType::INT32);
  tensor2.ref_dims().push_back(2);
  tensor2.ref_int32_data().push_back(10);
  tensor2.ref_int32_data().push_back(20);

  TensorProto& tensor3 = attribute.add_tensors();
  tensor3.set_name("tensor3");
  tensor3.set_data_type(TensorProto::DataType::INT64);
  tensor3.ref_dims().push_back(1);
  tensor3.ref_int64_data().push_back(10);

  TensorProto& tensor4 = attribute.add_tensors();
  tensor4.set_name("tensor4");
  tensor4.set_data_type(TensorProto::DataType::INT32);
  tensor4.ref_dims().push_back(1);
  tensor4.ref_int32_data().push_back(10);

  SerializeOptions options;
  {
    std::string serialized;
    utils::StringWriteStream stream;
    tensor2.SerializeToString(serialized);
    EXPECT_EQ(serialized.size(), tensor2.SerializeSize(stream, options));
  }
  {
    std::string serialized;
    utils::StringWriteStream stream;
    tensor3.SerializeToString(serialized);
    EXPECT_EQ(serialized.size(), tensor3.SerializeSize(stream, options));
  }
  {
    std::string serialized;
    utils::StringWriteStream stream;
    tensor4.SerializeToString(serialized);
    EXPECT_EQ(serialized.size(), tensor4.SerializeSize(stream, options));
  }
  {
    std::string serialized;
    utils::StringWriteStream stream;
    tensor1.SerializeToString(serialized);
    EXPECT_EQ(serialized.size(), tensor1.SerializeSize(stream, options));
  }
  {
    std::string serialized;
    utils::StringWriteStream stream;
    attribute.SerializeToString(serialized);
    EXPECT_EQ(serialized.size(), attribute.SerializeSize(stream, options));
  }
}

TEST(onnx2_proto, SerializeSize_ConsistencyAcrossTypes) {
  // Test with NodeProto
  NodeProto node;
  node.set_name("test_node");
  node.set_op_type("TestOp");
  node.add_input() = "input";
  node.add_output() = "output";

  std::string node_serialized;
  node.SerializeToString(node_serialized);
  utils::StringWriteStream node_stream;
  SerializeOptions options;
  EXPECT_EQ(node_serialized.size(), node.SerializeSize(node_stream, options));

  // Test with GraphProto
  GraphProto graph;
  graph.set_name("test_graph");
  NodeProto& graph_node = graph.add_node();
  graph_node.set_name("node_in_graph");

  std::string graph_serialized;
  graph.SerializeToString(graph_serialized);
  utils::StringWriteStream graph_stream;
  EXPECT_EQ(graph_serialized.size(), graph.SerializeSize(graph_stream, options));

  // Test with ModelProto
  ModelProto model;
  model.set_ir_version(7);
  model.set_producer_name("test_model");
  GraphProto& model_graph = model.add_graph();
  model_graph.set_name("graph_in_model");

  std::string model_serialized;
  model.SerializeToString(model_serialized);
  utils::StringWriteStream model_stream;
  EXPECT_EQ(model_serialized.size(), model.SerializeSize(model_stream, options));
}

TEST(onnx2_file, LoadOnnxFile_OldProtobuf) {
  namespace fs = std::filesystem;
  fs::path source_path = __FILE__;
  fs::path source_dir = source_path.parent_path();
  fs::path file_path = source_dir / ".." / ".." / "backend" / "test" / "data" / "node" / "test_ai_onnx_ml_binarizer" / "model.onnx";

  ModelProto model;
  utils::FileStream stream(file_path.string());
  onnx2::ParseOptions opts;
  model.ParseFromStream(stream, opts);

  utils::PrintOptions pr;
  std::string text = utils::join_string(model.PrintToVectorString(pr), "\n");
  EXPECT_NE(text.find("Binarizer"), std::string::npos);
}

TEST(onnx2_file, LoadOnnxFile_Expanded) {
  namespace fs = std::filesystem;
  fs::path source_path = __FILE__;
  fs::path source_dir = source_path.parent_path();
  fs::path file_path = source_dir / ".." / ".." / "backend" / "test" / "data" / "node" / "test_softmax_example_expanded" / "model.onnx";

  ModelProto model;
  utils::FileStream stream(file_path.string());
  onnx2::ParseOptions opts;
  model.ParseFromStream(stream, opts);

  utils::PrintOptions pr;
  std::string text = utils::join_string(model.PrintToVectorString(pr), "\n");
  EXPECT_NE(text.find("ReduceSum"), std::string::npos);
}

TEST(onnx2_file, LoadOnnxFile_ConstantAsString) {
  std::vector<uint8_t> data = {
      18,  3,   65,  65, 65,  26,  2,   78, 78, 34,  8, 67, 111, 110, 115, 116, 97,  110, 116, 42,  39,
      10,  5,   118, 97, 108, 117, 101, 42, 16, 8,   1, 16, 6,   58,  10,  255, 255, 255, 255, 255, 255,
      255, 255, 255, 1,  106, 3,   68,  79, 67, 160, 1, 4,  170, 1,   3,   82,  69,  70,  58,  1,   77};
  std::string data_str(data.begin(), data.end());
  EXPECT_EQ(data_str.size(), data.size());
  NodeProto node;
  node.ParseFromString(data_str);

  utils::PrintOptions pr;
  std::string text = utils::join_string(node.PrintToVectorString(pr), "\n");
  EXPECT_NE(text.find("Constant"), std::string::npos);
}

TEST(onnx2_proto, TensorProto_uint64) {
  TensorProto tensor = TensorProto();
  tensor.set_name("tensor");
  tensor.set_data_type(TensorProto::DataType::UINT64);
  tensor.ref_dims().push_back(2);
  tensor.ref_uint64_data().push_back(4);
  tensor.ref_uint64_data().push_back(5);

  SerializeOptions options;
  std::string serialized;
  tensor.SerializeToString(serialized);

  TensorProto t2 = TensorProto();
  ParseOptions parse_options;
  t2.ParseFromString(serialized, parse_options);

  EXPECT_EQ(t2.ref_name(), tensor.ref_name());
  EXPECT_EQ(t2.ref_data_type(), tensor.ref_data_type());
  EXPECT_EQ(t2.ref_dims().size(), tensor.ref_dims().size());
  EXPECT_EQ(t2.ref_uint64_data().size(), tensor.ref_uint64_data().size());
  EXPECT_EQ(t2.ref_uint64_data()[0], 4);
  EXPECT_EQ(t2.ref_uint64_data()[1], 5);
  utils::StringWriteStream stream;
  EXPECT_EQ(serialized.size(), tensor.SerializeSize(stream, options));
}

TEST(onnx2_proto, AttributeProto_float) {
  AttributeProto attribute = AttributeProto();
  attribute.set_name("attribute");
  attribute.set_type(AttributeProto::AttributeType::FLOAT);
  attribute.set_f(0.01f);

  SerializeOptions options;
  std::string serialized;
  attribute.SerializeToString(serialized);

  AttributeProto t2 = AttributeProto();
  ParseOptions parse_options;
  t2.ParseFromString(serialized, parse_options);

  EXPECT_EQ(t2.ref_name(), attribute.ref_name());
  EXPECT_EQ(t2.ref_type(), attribute.ref_type());
  EXPECT_EQ(t2.ref_f(), attribute.ref_f());
  utils::StringWriteStream stream;
  EXPECT_EQ(serialized.size(), attribute.SerializeSize(stream, options));
}

//

TEST(onnx2_proto, AttributeProto_TypeAttribute) {
  AttributeProto attribute;

  attribute.set_name("input_type");
  attribute.set_type(AttributeProto::AttributeType::TYPE_PROTO);

  TypeProto& type = attribute.add_tp();
  type.add_tensor_type().set_elem_type(1); // FLOAT
  TensorShapeProto& shape = type.ref_tensor_type().add_shape();
  TensorShapeProto::Dimension &dim1 = shape.add_dim();
  dim1.set_dim_value(3);
  TensorShapeProto::Dimension &dim2 = shape.add_dim();
  dim2.set_dim_param("N");

  EXPECT_EQ(attribute.ref_name(), "input_type");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::TYPE_PROTO);
  EXPECT_TRUE(attribute.has_tp());
  EXPECT_TRUE(attribute.ref_tp().has_tensor_type());
  EXPECT_EQ(attribute.ref_tp().ref_tensor_type().ref_elem_type(), 1);
  EXPECT_TRUE(attribute.ref_tp().ref_tensor_type().has_shape());
  EXPECT_EQ(attribute.ref_tp().ref_tensor_type().ref_shape().ref_dim().size(), 2);
  EXPECT_EQ(attribute.ref_tp().ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 3);
  EXPECT_EQ(attribute.ref_tp().ref_tensor_type().ref_shape().ref_dim()[1].ref_dim_param(), "N");
}

TEST(onnx2_proto, AttributeProto_TypesAttribute) {
  AttributeProto attribute;

  attribute.set_name("output_types");
  attribute.set_type(AttributeProto::AttributeType::TYPE_PROTO);

  // Premier type
  TypeProto& type1 = attribute.add_tp();
  type1.add_tensor_type().set_elem_type(1); // FLOAT
  TensorShapeProto& shape1 = type1.ref_tensor_type().add_shape();
  shape1.add_dim().set_dim_value(2);
  shape1.add_dim().set_dim_value(3);

  EXPECT_EQ(attribute.ref_name(), "output_types");
  EXPECT_EQ(attribute.ref_type(), AttributeProto::AttributeType::TYPE_PROTO);
  EXPECT_TRUE(attribute.ref_tp().has_tensor_type());
  EXPECT_EQ(attribute.ref_tp().ref_tensor_type().ref_elem_type(), 1);
  EXPECT_EQ(attribute.ref_tp().ref_tensor_type().ref_shape().ref_dim().size(), 2);
  EXPECT_EQ(attribute.ref_tp().ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 2);
}

TEST(onnx2_proto, AttributeProto_Serialization_TypeProto) {
  AttributeProto type_attr;
  type_attr.set_name("type_attr");
  type_attr.set_type(AttributeProto::AttributeType::TYPE_PROTO);

  TypeProto& type = type_attr.add_tp();
  type.add_tensor_type().set_elem_type(1); // FLOAT
  TensorShapeProto& shape = type.ref_tensor_type().add_shape();
  shape.add_dim().set_dim_value(4);
  shape.add_dim().set_dim_param("dynamic_dim");

  std::string serialized;
  type_attr.SerializeToString(serialized);

  AttributeProto deserialized;
  deserialized.ParseFromString(serialized);

  EXPECT_EQ(deserialized.ref_name(), "type_attr");
  EXPECT_EQ(deserialized.ref_type(), AttributeProto::AttributeType::TYPE_PROTO);
  EXPECT_TRUE(deserialized.has_tp());
  EXPECT_TRUE(deserialized.ref_tp().has_tensor_type());
  EXPECT_EQ(deserialized.ref_tp().ref_tensor_type().ref_elem_type(), 1);
  EXPECT_TRUE(deserialized.ref_tp().ref_tensor_type().has_shape());
  EXPECT_EQ(deserialized.ref_tp().ref_tensor_type().ref_shape().ref_dim().size(), 2);
  EXPECT_EQ(deserialized.ref_tp().ref_tensor_type().ref_shape().ref_dim()[0].ref_dim_value(), 4);
  EXPECT_EQ(deserialized.ref_tp().ref_tensor_type().ref_shape().ref_dim()[1].ref_dim_param(), "dynamic_dim");
}

//

TEST(onnx2_proto, TensorProto_DataLocation) {
  // Créer un TensorProto avec une localisation externe
  TensorProto tensor;
  tensor.set_name("external_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.ref_dims().push_back(2);
  tensor.ref_dims().push_back(3);

  // Par défaut, la localisation des données est ONNX2_DEFAULT
  EXPECT_EQ(tensor.ref_data_location(), TensorProto::DataLocation::DEFAULT);

  // Définir la localisation comme externe
  tensor.set_data_location(TensorProto::DataLocation::EXTERNAL);
  EXPECT_EQ(tensor.ref_data_location(), TensorProto::DataLocation::EXTERNAL);

  // Sérialiser et désérialiser
  std::string serialized;
  tensor.SerializeToString(serialized);

  TensorProto tensor2;
  tensor2.ParseFromString(serialized);

  // Vérifier que la localisation des données est préservée
  EXPECT_EQ(tensor2.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
  EXPECT_EQ(tensor2.ref_name(), "external_tensor");
  EXPECT_EQ(tensor2.ref_data_type(), TensorProto::DataType::FLOAT);
  EXPECT_EQ(tensor2.ref_dims().size(), 2);

  // Tester toutes les valeurs possibles de DataLocation
  TensorProto tensor3;
  tensor3.set_data_location(TensorProto::DataLocation::DEFAULT);
  EXPECT_EQ(tensor3.ref_data_location(), TensorProto::DataLocation::DEFAULT);

  tensor3.set_data_location(TensorProto::DataLocation::EXTERNAL);
  EXPECT_EQ(tensor3.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
}

TEST(onnx2_proto, TensorProto_ExternalData) {
  // Créer un TensorProto avec données externes
  TensorProto tensor;
  tensor.set_name("external_data_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.set_data_location(TensorProto::DataLocation::EXTERNAL);

  // Ajouter des informations sur les données externes
  StringStringEntryProto& entry1 = tensor.add_external_data();
  entry1.set_key("location");
  entry1.set_value("weights.bin");

  StringStringEntryProto& entry2 = tensor.add_external_data();
  entry2.set_key("offset");
  entry2.set_value("0");

  StringStringEntryProto& entry3 = tensor.add_external_data();
  entry3.set_key("length");
  entry3.set_value("1024");

  // Vérifier les entrées
  EXPECT_EQ(tensor.ref_external_data().size(), 3);
  EXPECT_EQ(tensor.ref_external_data()[0].ref_key(), "location");
  EXPECT_EQ(tensor.ref_external_data()[0].ref_value(), "weights.bin");
  EXPECT_EQ(tensor.ref_external_data()[1].ref_key(), "offset");
  EXPECT_EQ(tensor.ref_external_data()[1].ref_value(), "0");
  EXPECT_EQ(tensor.ref_external_data()[2].ref_key(), "length");
  EXPECT_EQ(tensor.ref_external_data()[2].ref_value(), "1024");

  // Sérialiser et désérialiser
  std::string serialized;
  tensor.SerializeToString(serialized);

  TensorProto tensor2;
  tensor2.ParseFromString(serialized);

  // Vérifier que les informations externes sont préservées
  EXPECT_EQ(tensor2.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
  EXPECT_EQ(tensor2.ref_external_data().size(), 3);
  EXPECT_EQ(tensor2.ref_external_data()[0].ref_key(), "location");
  EXPECT_EQ(tensor2.ref_external_data()[0].ref_value(), "weights.bin");
}

TEST(onnx2_proto, TensorProto_DataLocationPrintToVectorString) {
  utils::PrintOptions options;
  TensorProto tensor;
  tensor.set_name("external_print_tensor");
  tensor.set_data_type(TensorProto::DataType::FLOAT);
  tensor.set_data_location(TensorProto::DataLocation::EXTERNAL);

  StringStringEntryProto& entry = tensor.add_external_data();
  entry.set_key("location");
  entry.set_value("external_file.bin");

  // Générer la représentation textuelle
  std::vector<std::string> result = tensor.PrintToVectorString(options);
  ASSERT_FALSE(result.empty());

  // Vérifier que la sortie contient les informations de localisation des données
  bool foundDataLocation = false;
  bool foundExternalData = false;

  std::string serialized = utils::join_string(result, "\n");

  if (serialized.find("data_location:") != std::string::npos &&
      serialized.find(std::to_string(static_cast<int>(TensorProto::DataLocation::EXTERNAL))) !=
          std::string::npos) {
    foundDataLocation = true;
  }

  if (serialized.find("external_data") != std::string::npos &&
      serialized.find("location") != std::string::npos &&
      serialized.find("external_file.bin") != std::string::npos) {
    foundExternalData = true;
  }

  EXPECT_TRUE(foundDataLocation);
  EXPECT_TRUE(foundExternalData);
}

TEST(onnx2_proto, TensorProto_CopyFromWithDataLocation) {
  TensorProto source;
  source.set_name("source_external_tensor");
  source.set_data_type(TensorProto::DataType::FLOAT);
  source.set_data_location(TensorProto::DataLocation::EXTERNAL);

  StringStringEntryProto& entry = source.add_external_data();
  entry.set_key("location");
  entry.set_value("source_file.bin");

  // Copier les données vers une nouvelle instance
  TensorProto target;
  target.CopyFrom(source);

  // Vérifier que toutes les propriétés liées à la localisation des données sont copiées
  EXPECT_EQ(target.ref_name(), "source_external_tensor");
  EXPECT_EQ(target.ref_data_location(), TensorProto::DataLocation::EXTERNAL);
  EXPECT_EQ(target.ref_external_data().size(), 1);
  EXPECT_EQ(target.ref_external_data()[0].ref_key(), "location");
  EXPECT_EQ(target.ref_external_data()[0].ref_value(), "source_file.bin");
}
