// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include <ctype.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <strstream>

#include "onnx/onnx_pb.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
namespace Utils {

// Exception class used for handling parse errors

class ParseError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;

  ParseError(const std::string& message) : std::runtime_error(message) {}

  const char* what() const noexcept override {
    if (!expanded_message_.empty()) {
      return expanded_message_.c_str();
    }
    return std::runtime_error::what();
  }

  void AppendContext(const std::string& context) {
    expanded_message_ = context;
    // ONNX_NAMESPACE::MakeString(
    // std::runtime_error::what(), "\n\n==> Context: ", context);
  }

 private:
  std::string expanded_message_;
};

#define parse_error(...)                                                                                          \
  do {                                                                                                            \
    throw ParseError(ONNX_NAMESPACE::MakeString("[ParseError at position ", (next_ - start_), "]", __VA_ARGS__)); \
  } while (0)

class ParserBase {
 public:
  ParserBase(const std::string& str) : start_(str.data()), next_(str.data()), end_(str.data() + str.length()) {}

  ParserBase(const char* cstr) : start_(cstr), next_(cstr), end_(cstr + strlen(cstr)) {}

  void SkipWhiteSpace() {
    while ((next_ < end_) && (isspace(*next_)))
      next_++;
  }

  int NextChar(bool skipspace = true) {
    if (skipspace)
      SkipWhiteSpace();
    return (next_ < end_) ? *next_ : 0;
  }

  std::string ParseIdentifier() {
    SkipWhiteSpace();
    auto from = next_;
    if ((next_ < end_) && (isalpha(*next_) || (*next_ == '_'))) {
      next_++;
      while ((next_ < end_) && (isalnum(*next_) || (*next_ == '_')))
        next_++;
    }
    if (next_ == from)
      parse_error("Identifier expected but not found.");
    return std::string(from, next_ - from);
  }

  enum class TokenType { INT_LITERAL, FLOAT_LITERAL, STRING_LITERAL };

  struct Token {
    TokenType type;
    std::string value;
  };

  bool ParseIntValue(uint64_t& val) {
    SkipWhiteSpace();
    auto from = next_;
    while ((next_ < end_) && (isdigit(*next_)))
      next_++;
    if (next_ == from)
      return false;
    val = std::stol(std::string(from, next_ - from));
    return true;
  }

  Token ParseValue() {
    Token result;
    bool decimal_point = false;
    auto nextch = NextChar();
    auto from = next_;
    if (nextch == '"') {
      next_++;
      // TODO: Handle escape characters
      while ((next_ < end_) && (*next_ != '"')) {
        next_++;
      }
      next_++;
      result.type = TokenType::STRING_LITERAL;
      result.value = std::string(from + 1, next_ - from - 2); // skip enclosing quotes

    } else if ((isdigit(nextch) || (nextch == '-'))) {
      next_++;

      while ((next_ < end_) && (isdigit(*next_) || (*next_ == '.'))) {
        if (*next_ == '.') {
          if (decimal_point)
            break; // Only one decimal point allowed in numeric literal
          decimal_point = true;
        }
        next_++;
      }

      if (next_ == from)
        parse_error("Value expected but not found.");

      result.value = std::string(from, next_ - from);
      result.type = decimal_point ? TokenType::FLOAT_LITERAL : TokenType::INT_LITERAL;
    }
    return result;
  }

  bool Matches(char ch, bool skipspace = true) {
    if (skipspace)
      SkipWhiteSpace();
    if ((next_ < end_) && (*next_ == ch)) {
      ++next_;
      return true;
    }
    return false;
  }

  void Match(char ch, bool skipspace = true) {
    if (!Matches(ch, skipspace))
      parse_error("Expected character %c but not found", ch);
  }

  bool EndOfInput() {
    SkipWhiteSpace();
    return (next_ >= end_);
  }

 protected:
  const char* next_;
  const char* start_;
  const char* end_;
};

using IdList = google::protobuf::RepeatedPtrField<std::string>;

using NodeList = google::protobuf::RepeatedPtrField<NodeProto>;

using AttrList = google::protobuf::RepeatedPtrField<AttributeProto>;

class OnnxParser : public ParserBase {
 public:
  OnnxParser(const char* cstr) : ParserBase(cstr) {}

  void ParseIdList(IdList& idlist) {
    idlist.Clear();
    // Allow empty identifier (used for missing optional parameters) except
    // in special case. TODO: fix this.
    std::string id = ParseIdentifier();
    if (id.empty())
      return;
    idlist.Add(std::move(id));
    while (Matches(',')) {
      idlist.Add(ParseIdentifier());
    }
  }

  void Parse(TensorShapeProto& shape) {
    shape.clear_dim();
    do {
      auto ch = NextChar();
      uint64_t dimval;
      if (Matches('?')) {
        shape.add_dim();
      } else if (ParseIntValue(dimval)) {
        shape.add_dim()->set_dim_value(dimval);
      } else {
        shape.add_dim()->set_dim_param(ParseIdentifier());
      }
    } while (Matches(','));
  }

  void Parse(TypeProto& typeProto) {
    std::string id = ParseIdentifier();
    TensorProto_DataType dtype = TensorProto_DataType::TensorProto_DataType_UNDEFINED;
    if (TensorProto_DataType_Parse(id, &dtype)) {
      auto* tensortype = typeProto.mutable_tensor_type();
      tensortype->set_elem_type((int)dtype);
      tensortype->clear_shape();
      // Grammar:
      // FLOAT indicates scalar (rank 0)
      // FLOAT [] indicates unknown rank tensor
      // FLOAT [one-or-more-dimensions] indicates tensor of known rank > 0.
      if (Matches('[')) {
        if (!Matches(']')) {
          Parse(*tensortype->mutable_shape());
          Match(']');
        }
      } else {
        // Create shape with zero dimensions for scalar
        (void) (tensortype->mutable_shape());
      }
    } else
      parse_error("Unexpected type.");
  }

  void ParseSingleAttributeValue(AttributeProto& attr) {
    // Parse a single-value
    auto token = ParseValue();
    switch (token.type) {
      case TokenType::INT_LITERAL:
        attr.set_type(AttributeProto_AttributeType_INT);
        attr.set_i(std::stol(token.value));
        break;
      case TokenType::FLOAT_LITERAL:
        attr.set_type(AttributeProto_AttributeType_FLOAT);
        attr.set_f(static_cast<float>(std::stof(token.value)));
        break;
      case TokenType::STRING_LITERAL:
        attr.set_type(AttributeProto_AttributeType_STRING);
        attr.set_s(token.value);
        break;
      default:
        parse_error("Unexpected literal type.");
    }
  }

  void Parse(AttributeProto& attr) {
    attr.set_name(ParseIdentifier());
    Match('=');
    if (NextChar() == '[') {
      // Parse a list of values
      std::vector<Token> vals;
      Match('[');
      bool first_time = true;
      while (!Matches(']')) {
        AttributeProto nextval;
        ParseSingleAttributeValue(nextval);
        switch (nextval.type()) {
          case AttributeProto_AttributeType_INT:
            attr.set_type(AttributeProto_AttributeType_INTS);
            attr.add_ints(nextval.i());
            break;
          case AttributeProto_AttributeType_FLOAT:
            attr.set_type(AttributeProto_AttributeType_FLOATS);
            attr.add_floats(nextval.f());
            break;
          case AttributeProto_AttributeType_STRING:
            attr.add_strings(nextval.s());
            attr.set_type(AttributeProto_AttributeType_STRINGS);
            break;
          default:
            break;
        }
        (void)Matches(',');
      }
    } else {
      ParseSingleAttributeValue(attr);
    }
  }

  void Parse(AttrList& attrlist) {
    attrlist.Clear();
    if (Matches('{')) {
      while (!Matches('}')) {
        Parse(*attrlist.Add());
        (void)Matches(','); // skip optional comma if present
      }
    }
  }

  void Parse(NodeProto& node) {
    ParseIdList(*node.mutable_output());
    Match('=');
    node.set_op_type(ParseIdentifier());
    Parse(*node.mutable_attribute());
    Match('(');
    ParseIdList(*node.mutable_input());
    Match(')');
  }

  void Parse(NodeList& nodelist) {
    nodelist.Clear();
    Match('{');
    while (!Matches('}')) {
      Parse(*nodelist.Add());
      (void)Matches(';'); // skip optional semicolon if present
    }
  }

  void Parse(GraphProto& graph) {
    graph.set_name(ParseIdentifier());
    Match('(');
    Match(')');
    Parse(*graph.mutable_node());
  }

  template <typename T>
  static void Parse(T& parsedData, const char* input) {
    OnnxParser parser(input);
    parser.Parse(parsedData);
  }
}; // namespace Utils

} // namespace Utils
} // namespace ONNX_NAMESPACE