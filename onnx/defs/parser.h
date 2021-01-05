// Copyright (c) ONNX Project Contributors.
// Licensed under the MIT license.

#pragma once

#include <ctype.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <strstream>
#include <unordered_map>

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

class KeyWordMap {
 public:
  enum class KeyWord {
    NONE,
    IR_VERSION,
    OPSET_IMPORT,
    PRODUCER_NAME,
    PRODUCER_VERSION,
    DOMAIN_KW,
    MODEL_VERSION,
    DOC_STRING,
    METADATA_PROPS
  };

  KeyWordMap() {
    map_["ir_version"] = KeyWord::IR_VERSION;
    map_["opset_import"] = KeyWord::OPSET_IMPORT;
    map_["producer_name"] = KeyWord::PRODUCER_NAME;
    map_["producer_version"] = KeyWord::PRODUCER_VERSION;
    map_["domain"] = KeyWord::DOMAIN_KW;
    map_["model_version"] = KeyWord::MODEL_VERSION;
    map_["doc_string"] = KeyWord::DOC_STRING;
    map_["metadata_props"] = KeyWord::METADATA_PROPS;
  }

  static const std::unordered_map<std::string, KeyWord>& Instance() {
    static KeyWordMap instance;
    return instance.map_;
  }

  static KeyWord Lookup(const std::string& id) {
    auto it = Instance().find(id);
    if (it != Instance().end())
      return it->second;
    return KeyWord::NONE;
  }

 private:
  std::unordered_map<std::string, KeyWord> map_;
};

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

  KeyWordMap::KeyWord ParseKeyWord() {
    return KeyWordMap::Lookup(ParseIdentifier());
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

  uint64_t ParseIntValue() {
    uint64_t result;
    if (!ParseIntValue(result))
      parse_error("Integer value expected but not found.");
    return result;
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

  std::string ParseString() {
    auto token = ParseValue();
    if (token.type != TokenType::STRING_LITERAL)
      parse_error("String value expected, but not found.");
    return token.value;
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
  const char* start_;
  const char* next_;
  const char* end_;
};

using IdList = google::protobuf::RepeatedPtrField<std::string>;

using NodeList = google::protobuf::RepeatedPtrField<NodeProto>;

using AttrList = google::protobuf::RepeatedPtrField<AttributeProto>;

using ValueInfoList = google::protobuf::RepeatedPtrField<ValueInfoProto>;

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
    *idlist.Add() = id;
    while (Matches(',')) {
      *idlist.Add() = ParseIdentifier();
    }
  }

  void Parse(TensorShapeProto& shape) {
    shape.clear_dim();
    do {
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
        (void)(tensortype->mutable_shape());
      }
    } else
      parse_error("Unexpected type.");
  }

  void Parse(ValueInfoProto& valueinfo) {
    Parse(*valueinfo.mutable_type());
    valueinfo.set_name(ParseIdentifier());
  }

  void Parse(ValueInfoList& vilist) {
    vilist.Clear();
    Match('(');
    while (!Matches(')')) {
      Parse(*vilist.Add());
      (void)Matches(','); // skip optional comma if present
    }
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
    if (Matches('<')) {
      while (!Matches('>')) {
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
    Parse(*graph.mutable_input());
    Match('=');
    Match('>', false);
    Parse(*graph.mutable_output());
    Parse(*graph.mutable_node());
  }

  void Parse(ModelProto& model) {
    if (Matches('<')) {
      while (!Matches('>')) {
        auto keyword = ParseKeyWord();
        Match(':');
        switch (keyword) {
          case KeyWordMap::KeyWord::IR_VERSION:
            model.set_ir_version(ParseIntValue());
            break;
          case KeyWordMap::KeyWord::OPSET_IMPORT: {
            auto& imports = *model.mutable_opset_import();
            Match('[');
            while (!Matches(']')) {
              auto* import = imports.Add();
              import->set_domain(ParseString());
              Match(':');
              import->set_version(ParseIntValue());
            }
            break;
          }
          case KeyWordMap::KeyWord::PRODUCER_NAME:
            model.set_producer_name(ParseString());
            break;
          case KeyWordMap::KeyWord::PRODUCER_VERSION:
            model.set_producer_version(ParseString());
            break;
          case KeyWordMap::KeyWord::DOMAIN_KW:
            model.set_domain(ParseString());
            break;
          case KeyWordMap::KeyWord::MODEL_VERSION:
            model.set_model_version(ParseIntValue());
            break;
          case KeyWordMap::KeyWord::DOC_STRING:
            model.set_doc_string(ParseString());
            break;
          case KeyWordMap::KeyWord::METADATA_PROPS: {
            auto& metadata_props = *model.mutable_metadata_props();
            Match('[');
            while (!Matches(']')) {
              auto* metadata = metadata_props.Add();
              metadata->set_key(ParseString());
              Match(':');
              metadata->set_value(ParseString());
            }
            break;
          }
          default:
            parse_error("Unhandled keyword.");
        }
      }
    }
    Parse(*model.mutable_graph());
  }

  template <typename T>
  static void Parse(T& parsedData, const char* input) {
    OnnxParser parser(input);
    parser.Parse(parsedData);
  }
};

} // namespace Utils
} // namespace ONNX_NAMESPACE