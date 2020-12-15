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

  Token ParseValue() {
    bool decimal_point = false;
    SkipWhiteSpace();
    auto from = next_;
    if ((next_ < end_) && (isdigit(*next_) || (*next_ == '-'))) {
      next_++;

      while ((next_ < end_) && (isdigit(*next_) || (*next_ == '.'))) {
        if (*next_ == '.') {
          if (decimal_point)
            break; // Only one decimal point allowed in numeric literal
          decimal_point = true;
        }
        next_++;
      }
    }
    if (next_ == from)
      parse_error("Value expected but not found.");
    Token result;
    result.value = std::string(from, next_ - from);
    result.type = decimal_point ? TokenType::FLOAT_LITERAL : TokenType::INT_LITERAL;
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

  void ParseAttribute(AttributeProto& attr) {
    attr.set_name(ParseIdentifier());
    Match('=');
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
      default:
        parse_error("Unexpected literal type.");
    }
  }

  void ParseOptionalAttributeList(AttrList& attrlist) {
    attrlist.Clear();
    if (Matches('{')) {
      while (!Matches('}')) {
        ParseAttribute(*attrlist.Add());
        (void)Matches(','); // skip optional comma if present
      }
    }
  }

  static void Parse(AttrList& attrlist, const char* input) {
    OnnxParser parser(input);
    parser.ParseOptionalAttributeList(attrlist);
  }

  void ParseNode(NodeProto& node) {
    ParseIdList(*node.mutable_output());
    Match('=');
    node.set_op_type(ParseIdentifier());
    ParseOptionalAttributeList(*node.mutable_attribute());
    Match('(');
    ParseIdList(*node.mutable_input());
    Match(')');
  }

  static void Parse(NodeProto& node, const char* input) {
    OnnxParser parser(input);
    parser.ParseNode(node);
  }

  void ParseNodeList(NodeList& nodelist) {
    nodelist.Clear();
    Match('{');
    while (!Matches('}')) {
      ParseNode(*nodelist.Add());
      (void)Matches(';'); // skip optional semicolon if present
    }
  }

  static void Parse(NodeList& node, const char* input) {
    OnnxParser parser(input);
    parser.ParseNodeList(node);
  }
};

} // namespace Utils
} // namespace ONNX_NAMESPACE